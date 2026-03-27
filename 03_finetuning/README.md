# 03 — Fine-tuning TabPFN for time series classification

This module adapts the pretrained **TabPFN v2.5** tabular foundation model
(Hollmann et al., 2025) for **multivariate time series classification** through
transfer learning.  The core idea is to repurpose TabPFN's powerful
between-sample attention mechanism — originally designed for tabular rows — to
operate on **temporal patch tokens** extracted from time series, inspired by the
patch-based tokenisation of **PatchTST** (Nie et al., 2023).

---

## 1  Background: how standard TabPFN v2.5 works

TabPFN v2.5 is a **Prior-Data Fitted Network**: a transformer trained entirely
on synthetic tabular datasets sampled from a Bayesian DAG prior, so that at
inference time it can classify new tabular tasks in a single forward pass
(in-context learning), without gradient-based training.

### 1.1  Input representation and feature limits

TabPFN accepts a flat 2-D matrix `(n_samples, n_features)`.
The v2.5 inference configuration sets:

| Limit | Value |
|-------|-------|
| `MAX_NUMBER_OF_FEATURES` | 2 000 |
| `MAX_NUMBER_OF_SAMPLES` | 50 000 |
| `max_features_per_estimator` | 500 |

The model formally accepts up to **2 000 features**, but each ensemble
estimator only sees a **random subset of 500 features** (sampled without
replacement during `fit`).  With 8 estimators, different subsets are drawn for
each, providing combined coverage of the full feature space.

Internally, the model groups consecutive features into **tokens** of size
`features_per_group` (3 in the v2.5 checkpoint).  Each token is passed through
an **encoder** pipeline that produces a 192-dimensional embedding.  A special
**y-token** (carrying the label for train rows, or a NaN placeholder for test
rows) is appended to each sample's token sequence.

### 1.2  Preprocessing pipeline (per estimator)

Before the data reaches the transformer, each ensemble estimator applies
its own independent sklearn-based preprocessing pipeline.  v2.5 defines **two
preprocessor recipes** (see `preprocessing/presets.py`):

| Recipe | Name | Scaler | Global transform | Categoricals |
|--------|------|--------|-----------------|--------------|
| A | `squashing_scaler_default` | SquashingScaler | SVD (quarter-components) | Ordinal, common categories shuffled |
| B | `none` | — | — | Numeric |

With `n_estimators=8`, recipes are **balanced**: estimators 0–3 use recipe A,
estimators 4–7 use recipe B.

The full pipeline for each estimator (in order):

1. **Feature subsampling** — if `n_features > 500`, sample 500 columns without
   replacement (different random subset per estimator).

2. **SquashingScaler** (recipe A only) — per-column robust scaling using the
   IQR (`q25`–`q75`), followed by a **soft clip**:
   `x_out = z / √(1 + (z/B)²)` with `B = 3.0`.  NaN is preserved; ±inf is
   clamped to ±B.  This bounds the data to approximately [-3, +3] while
   keeping the mapping smooth and differentiable.

3. **SVD** (recipe A only) — `FeatureUnion([passthrough, TruncatedSVD(…)])`.
   The SVD branch computes `n_components = max(1, min(n_train//10+1, n_feats//4))`
   components (so up to ~125 for 500 features with enough samples).  The
   result is the **original columns concatenated with SVD columns** — it
   *adds* dimensionality rather than replacing the input.  Each estimator
   uses a different random state for the SVD.

4. **Categorical encoding** — ordinal encoding for detected categoricals
   (recipe A) or plain numeric pass-through (recipe B).

5. **Fingerprint feature** — one extra column per row: a SHA-256 hash of the
   row's bytes mapped to `[0, 1)`.  Helps the model disambiguate duplicate
   rows via attention.

6. **Feature shuffle** (`FEATURE_SHIFT_METHOD="shuffle"`) — a full random
   column permutation, drawn independently per estimator.  This makes the
   ensemble approximately invariant to the original column ordering.

7. **Class permutation** (`CLASS_SHIFT_METHOD="shuffle"`) — a random
   permutation of class labels applied to `y_train` before the forward pass.
   The inverse permutation is applied to the output logits, so the final
   predictions are in the original label space.  Each estimator receives a
   different class permutation.

### 1.3  Encoder pipeline (SequentialEncoder)

After preprocessing, each token (group of `features_per_group` = 3 columns)
passes through the following encoder steps **in order**:

| # | Step | Purpose |
|---|------|---------|
| 1 | `RemoveEmptyFeaturesEncoderStep` | Drop all-NaN or constant columns |
| 2 | `NanHandlingEncoderStep` | Replace NaN/inf with per-feature train means; emit a binary indicator channel per feature (doubles width from `n` to `2n`) |
| 3 | `VariableNumFeaturesEncoderStep` *(NaN indicators)* | Pad/scale the indicator channels to fixed width |
| 4 | `InputNormalizationEncoderStep` | Per-position z-score normalisation using **train-only** means and stds; clip to ±100 |
| 5 | `VariableNumFeaturesEncoderStep` *(main)* | Pad/scale the main values to fixed `num_features`; optionally scale by number of used features |
| 6 | `MLP` or `Linear` projection | Map the concatenated `[main, indicators]` (width = `2 × features_per_group = 6`) to `emsize = 192` |

### 1.4  Feature embeddings (COL_EMBEDDING)

After encoding, each token receives a **pseudo-random embedding** that gives it
a unique "identity".  TabPFN uses the `subspace` mode with `seed=42`:

1. A fixed **2000 × 48** table of pseudo-random vectors (`COL_EMBEDDING`) is
   loaded from a shipped `.pt` file.
2. A pretrained `nn.Linear(48, 192)` projects each 48-dim vector to the model
   dimension.
3. The projected embedding is **added** to the encoder output:
   `x += emb[token_idx]`.

Each token receives a **different** embedding vector, so the transformer can
learn that "token 0 carries different information than token 5".  These
embeddings are **not** ordinal positional encodings — they do not encode
"first", "second", "third" — but they do give each token a distinguishable
identity.

### 1.5  Transformer backbone

The `PerFeatureTransformer` has **24 layers** (`emsize=192`, `nhead=3`,
`nhid=384`).  Each layer applies:

1. **Between-feature attention** — across the token (feature-group) dimension.
   All tokens within one sample attend to each other (including the y-token).
2. **Between-item attention** — across the sample (row) dimension.  This is
   what enables TabPFN's in-context learning: train and test rows attend to
   each other, allowing the model to "learn" from the training examples.
3. **MLP** feed-forward block with post-norm.

The final representation of the **y-token** for each test sample is passed
through a decoder head to produce class logits.

### 1.6  Ensembling and final prediction

With `n_estimators=8`, each estimator independently produces logits.  The
final prediction aggregates them:

1. **Undo class permutation**: each estimator's logits are re-ordered to the
   original label space.
2. **Temperature scaling**: logits are divided by `T = 0.9`.
3. **Softmax**: convert to probabilities.
4. **Average**: probabilities are averaged across all 8 estimators.

Because each estimator sees a **different random 500-feature subset**, a
different column permutation, a different class permutation, and half use
SquashingScaler + SVD while the other half don't, the ensemble acts as a
**diverse committee** that covers different views of the data.

### 1.7  Motivation for our adaptation

Standard TabPFN can already handle flattened time series reasonably well —
the model learns correlations between features through **in-context learning**
(attending across all training and test samples), and each feature-group token
receives a unique embedding that allows the transformer to differentiate them.

However, standard TabPFN groups **3 arbitrary consecutive columns** into each
token.  When a time series `(n, m, T)` is flattened to `(n, m×T)`, this means
that a token might contain timestep 7 of channel 0, timestep 8 of channel 0,
and timestep 9 of channel 0 — or, depending on alignment, it might straddle
channel boundaries.  The grouping is **arbitrary** and does not exploit the
strong **local temporal correlation** that exists in time series: nearby
timesteps within a channel are typically highly correlated and carry structured
patterns (trends, slopes, oscillations, peaks) that are best captured jointly.

Our key observation is that time series have **much stronger local structure**
than generic tabular data.  Inspired by **PatchTST** (Nie et al., 2023), we
group **16 consecutive timesteps from a single channel** into one token.  This
allows the encoder to extract meaningful local features (e.g. "this segment is
rising", "there is a peak here") that a 3-feature token cannot capture.

Additionally, TabPFN's per-position `InputNormalizationEncoderStep` z-scores
each column independently, which can wash out relative amplitude differences
between timesteps within a token.  We replace this with a **per-channel
normalisation** that preserves temporal dynamics within patches.

Finally, the **500-feature subsampling** means that for a series with
`m × T = 2000`, each estimator only sees 25% of the timesteps, discarding
temporal context.  Our overlap-patch approach avoids subsampling entirely,
ensuring every timestep contributes to every forward pass.

---

## 2  Our adaptation: temporal patching + global encoders

### 2.1  Inspiration: PatchTST

PatchTST (Nie et al., 2023) demonstrated that splitting a time series into
**overlapping fixed-length patches** and feeding each patch as a single token
to a transformer achieves strong performance on long-horizon forecasting.
Each patch preserves **local temporal structure** — trends, slopes, oscillations
— that would be diluted if individual timesteps were treated as independent
features.  The key insight is that temporal locality provides a strong
inductive bias: 16 adjacent timesteps are far more correlated with each other
than 16 random columns in a tabular dataset, so grouping them together lets
the encoder extract richer per-token representations.

We apply this principle to TabPFN's `features_per_group` mechanism:
instead of grouping 3 arbitrary features per token, we group **16 consecutive
timesteps** from a single channel into one token, with an overlap of 4
timesteps between adjacent patches (`window=16, stride=12`).

### 2.2  Transfer learning strategy

We perform **transfer learning** from the pretrained TabPFN v2.5 checkpoint:

| Component | Status | Rationale |
|-----------|--------|-----------|
| Patch encoder (MLP) | **Fresh** (Xavier init) | New input semantics: 16 timesteps instead of 3 tabular features |
| Global Conv1D encoder | **Fresh** (Xavier init) | Entirely new module |
| `feature_positional_embedding_embeddings` | **Pretrained** (frozen) | The `nn.Linear(48→192)` projection is task-agnostic |
| 24 transformer layers | **Pretrained** | Between-sample attention transfers directly to time series classification |
| y_encoder, decoder | **Pretrained** | Classification head is task-agnostic |

The pretrained transformer layers are the key asset: they already know how to
perform in-context classification by attending across samples.  Our adaptation
only changes **how raw input is tokenised**, not the classification mechanism
itself.

### 2.3  Architecture changes vs. vanilla TabPFN

#### `features_per_group`: 3 → 16

Each token now represents **1 channel × 16 timesteps** (rather than 3
arbitrary features).  For a series with `m` channels and `T` timesteps, after
overlap expansion we obtain `m × G` patch tokens, where
`G = ⌊(T_pad − 16) / 12⌋ + 1`.

#### Patch encoder: MLP(32 → 96 → GELU → 192)

The pretrained linear encoder (6 → 192) is replaced with a fresh 2-layer MLP:

```
[16 values + 16 NaN indicators] = 32 inputs
  → Linear(32, 96)  → GELU
  → Linear(96, 192)           (no bias, Xavier init)
```

This is constructed via TabPFN's `get_encoder` pipeline, which still includes:
- `NanHandlingEncoderStep` — replaces NaN with train means, emits indicators
- `VariableNumFeaturesEncoderStep` — pads/scales by used features

**Critically, `normalize_x=False`**: TabPFN's default `InputNormalizationEncoderStep`
is **disabled**.  In the tabular setting, per-position normalisation is
harmless because columns are independent.  But in a temporal patch, the 16
timesteps are **ordered and correlated** — per-position z-scoring would
independently shift each timestep, destroying trends and slopes within the
patch.

#### Per-channel normalisation (replaces per-position normalisation)

Instead of TabPFN's per-position z-scoring, we apply a single
**per-channel normalisation** step *before* overlap expansion:

- For each channel `j`, compute `μ_j` and `σ_j` over all `n_train × T` values
  (ignoring NaN).
- Normalise: `x[:, j, :] = (x[:, j, :] − μ_j) / σ_j` for both train and test.

This preserves temporal dynamics within patches (a rising trend from −1 to +1
stays a rising trend) while removing amplitude differences between channels and
datasets.  Both the patch encoder and global encoder see the same normalised
data.

#### Global Conv1D encoder

In addition to the patch tokens, we extract **global summary tokens** from
the full (non-patched) series using multi-scale 1-D convolutions, inspired by
the multi-scale random convolutional kernels of ROCKET (Dempster et al., 2020):

| Kernel size | 3 | 7 | 9 | 11 |
|-------------|---|---|---|---|
| Conv1d(2→192) + GELU | ✓ | ✓ | ✓ | ✓ |

For each kernel:
1. Input: `[normalised_value, nan_indicator]` — 2 channels per feature channel.
2. Apply `Conv1d(2, 192, k, padding=k//2)` + GELU → `(n×m, 192, T')`.
3. **Mean pool** and **max pool** over `T` → 2 vectors of dim 192 per channel.
4. If multivariate (`m > 1`): **mean pool** and **max pool** over `m`.

This produces:
- **8 global tokens** for univariate series (4 kernels × 2 T-pools)
- **16 global tokens** for multivariate series (4 kernels × 2 T-pools × 2 m-pools)

These tokens capture multi-scale summary statistics of the entire series that
complement the local patch tokens.

#### Token injection and embeddings

Global tokens are **concatenated** with patch tokens before TabPFN's
`add_embeddings` step (via a monkey-patched wrapper).  The combined token
sequence is:

```
[patch_0, patch_1, ..., patch_{m×G−1}, global_0, ..., global_{7 or 15}]
```

**All tokens — patch and global — receive unique COL_EMBEDDING vectors** from
TabPFN's pretrained `subspace` embedding system (seed=42).  The projection
`nn.Linear(48→192)` is pretrained and frozen.  This means:

- Each token gets a unique "identity" embedding, so the transformer can
  distinguish patch 0 from patch 5 and from global token 3.
- No explicit temporal position encoding is added — the pretrained
  pseudo-random embeddings serve as token identifiers, and the model learns
  the meaning of each position through in-context learning.

### 2.4  Parameter counts

| Group | Parameters | Status |
|-------|-----------|--------|
| Patch encoder (MLP 32→96→192) | ~21,500 | Fresh |
| Global Conv1D encoder (4 kernels) | ~12,300 | Fresh |
| **Total fresh** | **~33,800** | Trained from scratch |
| Transformer (24 layers) + embeddings + decoder | ~10,717,000 | Pretrained |
| **Total** | **~10,751,000** | |

Only **0.31%** of parameters are fresh; the rest are pretrained.

---

## 3  Preprocessing: TabPFN standard vs. ours

### 3.1  Standard TabPFN preprocessing (bypassed)

Standard TabPFN v2.5 applies the following sklearn-based preprocessing per
estimator (see §1.2): feature subsampling to 500, optional SquashingScaler,
optional SVD (as FeatureUnion with passthrough), categorical encoding,
fingerprint feature, feature shuffle, and class permutation.

### 3.2  Our preprocessing (direct tensor path)

All of TabPFN's sklearn preprocessing is **entirely bypassed**.  We use
`clf.fit_from_preprocessed()` with a dummy `PreprocessorConfig("none")` and
`FEATURE_SHIFT_METHOD=None`.

Our complete preprocessing pipeline is:

1. **Per-channel normalisation** — `per_channel_normalize()` using train-only
   statistics (mean=0, std=1 per channel).
2. **Overlap expansion** — `pad_and_expand_overlap()` (window=16, stride=12):
   extracts overlapping patches and flattens to `(n, m × G × 16)`.
3. **Global token computation** — `set_global_input()`: feeds the normalised
   3D data through the Conv1D encoder to produce 8 or 16 global tokens.
4. **NaN handling** — built into the patch encoder (`NanHandlingEncoderStep`):
   replaces NaN with train means and emits indicator channels.
5. **No** feature subsampling, SVD, squashing scaler, polynomial features,
   or fingerprint features.

---

## 4  Training: two-phase fine-tuning

Training uses a **3-GPU Slurm pipeline** where each worker runs on a separate
GPU:

| GPU | Worker | Role |
|-----|--------|------|
| 0 | `worker_generator.py` | Generates synthetic dataset batches (1/4 original + 3/4 augmented) |
| 1 | `worker_trainer_v2.py` | Trains the model, saves `last.pt` after every step |
| 2 | `worker_evaluator_v2.py` | Polls `last.pt`, evaluates on synthetic + real data, promotes to `best.pt` |

### 4.1  Phase 1: encoder warm-up (`step1_encoder.sbatch`)

**Trainable**: patch encoder + global Conv1D encoder (~33,800 params).
**Frozen**: embedding projection, all 24 transformer layers, y_encoder, decoder.

| Parameter | Default |
|-----------|---------|
| Learning rate | 1e-4 |
| Warmup (constant LR) | 1,500 steps |
| Cosine annealing | 3,000 steps → 1e-5 |
| Total steps | 4,500 |
| Batch size | 92 datasets/step |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Optimiser | AdamW |

**Rationale**: the fresh encoders must learn to produce embeddings in the right
distribution for the pretrained transformer layers.  Unfreezing everything from
the start would cause the random encoder outputs to corrupt the pretrained
weights via large gradients.

### 4.2  Phase 2: full fine-tune (`step2_finetune.sbatch`)

**Trainable**: all parameters (nothing frozen).
**Starts from**: `phase1/best.pt` (weights only; optimizer and scheduler reset).

| Parameter | Default |
|-----------|---------|
| Learning rate | 1e-5 (constant) |
| Total steps | 1,000 |
| Batch size | 92 datasets/step |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Optimiser | AdamW |

**Rationale**: with the encoders already warm, a small LR allows the
transformer layers to adapt their attention patterns to the new temporal token
layout without catastrophic forgetting of the pretrained in-context learning
ability.

### 4.3  LR schedule

Both phases use a **constant-then-cosine** schedule:

```
lr(step) = lr_max                                  if step < warmup_const
           cosine_decay(lr_max → lr_min, step)     if step ≥ warmup_const
```

For Phase 2 with `WARMUP_CONST = N_STEPS = 1000`, the LR is effectively
constant throughout training.

### 4.4  Training step mechanics

Each training step processes one batch of ~92 synthetic datasets:

1. For each dataset in the batch:
   a. Reshape to 3D `(n, m, T)`, per-channel normalise.
   b. Overlap-expand to `(n, m × G × 16)` flat features.
   c. Compute global tokens via `set_global_input()`.
   d. Forward pass → logits → cross-entropy loss.
   e. Backward pass → accumulate gradients.
2. Average accumulated gradients over valid datasets.
3. Clip gradient norm to 1.0.
4. Optimizer step + scheduler step.
5. Save `last.pt` atomically.

### 4.5  Synthetic data batches and training-time augmentation

Each batch contains `B/4` original synthetic datasets and `3B/4` augmented
copies (see `02_synthetic_data/` for the generation pipeline).  The generator
throttles output if more than 50 unprocessed `.npz` files accumulate,
preventing disk exhaustion.

The augmentation pipeline is critical for **training the model to be invariant
to the same transformations that the ensemble will apply at inference time**.
Each augmented copy independently applies (see `02_synthetic_data/augmentation.py`):

1. **Channel permutation** — random permutation of the `m` feature channels.
   This teaches the model that channel ordering carries no intrinsic meaning,
   analogous to TabPFN's feature shuffling across ensemble members.
2. **Class label permutation** — random relabelling of the `n_classes` target
   labels.  Prevents the model from associating any meaning with specific
   class indices, analogous to TabPFN's class shuffling.
3. **Per-channel value transforms** — each channel independently receives one
   of: identity (50%), log, exp, squashing scaler (IQR + soft clip), kernel
   density integral, or Kumaraswamy CDF warp.  This exposes the model to
   diverse amplitude distributions, analogous to seeing both squashed and
   non-squashed data across TabPFN's ensemble members.
4. **Temporal granularity transforms** — pooling (mean/max/min over sliding
   windows) or step-repeat (each timestep repeated `n` times), applied at
   the dataset level.  This teaches invariance to temporal resolution changes.
5. **Intentional missing values** — random NaN injection (low probability per
   channel), so the model learns to use the `NanHandlingEncoderStep`
   indicators effectively.
6. **Group-size padding** — right zero-pad `T` to a multiple of 8.

Because the model is trained on data that has been **randomly channel-shuffled
and class-permuted** in 75% of training examples, at inference time the
ensemble can apply the same operations (different channel permutation and
class permutation per iteration) and the model produces consistent predictions
regardless of the ordering.

---

## 5  Inference and ensembling

### 5.1  Single forward pass (`forward_single_dataset`)

Used during training and evaluation.  Bypasses all sklearn preprocessing:

1. `per_channel_normalize()` — train-only statistics.
2. `pad_and_expand_overlap()` — overlapping patch extraction.
3. `set_global_input()` — compute and store global Conv1D tokens.
4. `clf.fit_from_preprocessed()` + `clf.forward()` — direct tensor path.
5. `F.cross_entropy()` on raw logits.

### 5.2  Ensemble evaluation (`evaluate_ensemble`)

For final evaluation, we run multiple iterations (typically 1 or 8) and
average their predictions.  Each iteration creates a **different view** of
the same dataset, mirroring the diversity mechanisms of TabPFN's standard
ensemble.

**Step-by-step procedure for each iteration `it`:**

1. **Channel permutation** — draw a random permutation of `[0, 1, ..., m−1]`
   and reorder all `m` channels accordingly (same permutation applied to both
   train and test).  This is analogous to TabPFN's per-estimator feature
   shuffle (`FEATURE_SHIFT_METHOD="shuffle"`).  Because each iteration sees a
   different channel ordering, the pseudo-random COL_EMBEDDING assigned to
   each patch position changes, giving the transformer a different "view" of
   which channel maps to which token identity.

2. **Class permutation** — draw a random permutation of `[0, 1, ..., n_classes−1]`
   and relabel `y_train`.  This is analogous to TabPFN's per-estimator class
   shuffle (`CLASS_SHIFT_METHOD="shuffle"`).

3. **Optional squashing** (odd iterations only) — apply a **temporal squashing
   scaler**: per-channel robust scaling (median + IQR from train) followed by
   a soft clip `x / √(1 + (x/3)²)`.  This is analogous to alternating between
   TabPFN's two preprocessor recipes: estimators 0–3 use SquashingScaler + SVD,
   estimators 4–7 use no preprocessing.  In our ensemble, even iterations see
   raw normalised data and odd iterations see squashed data.

4. **Per-channel normalisation** — `per_channel_normalize()` using train-only
   statistics (applied after the optional squashing step).

5. **Overlap expansion** — `pad_and_expand_overlap()` extracts overlapping
   patches from the (possibly squashed, channel-permuted) flat data.

6. **Global token computation** — `set_global_input()` feeds the normalised
   3D data through the Conv1D encoder.

7. **Forward pass** — the model receives `[train_rows; test_rows]` and
   produces logits for the test rows.

8. **Undo class permutation** — the logits are re-ordered by indexing with
   the class permutation: `logits[:, class_perm]`, mapping predictions back
   to the original label space.

9. **Temperature scaling + softmax** — `softmax(logits / 0.9)`, matching
   TabPFN's default temperature of 0.9.

10. **Accumulate** — probabilities are summed across iterations.

**Final prediction** = sum of probabilities / number of valid iterations.

**Summary of correspondences with TabPFN's standard ensemble:**

| TabPFN ensemble mechanism | Our ensemble mechanism |
|--------------------------|----------------------|
| Feature subsampling (500 of 2000) | Not needed — all patch tokens used every iteration |
| Random column permutation per estimator | Random channel permutation per iteration |
| Random class permutation per estimator | Random class permutation per iteration |
| 4 estimators with SquashingScaler + SVD | Odd iterations with temporal squashing scaler |
| 4 estimators with no preprocessing | Even iterations with raw normalised data |
| Fingerprint feature | Not used |
| Temperature = 0.9, average probabilities | Temperature = 0.9, average probabilities |

The key difference is that TabPFN must **subsample features** because each
estimator is limited to 500 columns, relying on the ensemble to cover the full
feature space.  Our model processes **all patch tokens in every iteration** —
the ensemble diversity comes purely from channel/class permutations and
preprocessing variants, not from feature dropout.

### 5.3  Evaluation labels

In final evaluation (`04_evaluation/`), our model is compared with:

| Label | Description |
|-------|-------------|
| **e1** | Standard TabPFN, 1 estimator (precomputed baseline) |
| **e8** | Standard TabPFN, 8 estimators (precomputed baseline) |
| **D_e1** | Our finetuned model, 1 ensemble iteration |
| **D_e8** | Our finetuned model, 8 ensemble iterations |
| **HC2** | HIVE-COTE 2.0 SOTA reference |

Fair comparisons: D_e1 vs e1, D_e8 vs e8.

---

## 6  File reference

| File | Role |
|------|------|
| `model.py` | Model architecture: `build_overlap_model()`, overlap expansion, `GlobalConvEncoder`, `per_channel_normalize`, global token injection |
| `inference.py` | Forward pass (`forward_single_dataset`), ensemble evaluation (`evaluate_ensemble`), temporal squashing scaler |
| `worker_trainer_v2.py` | Training loop with two-phase schedule, gradient accumulation, atomic checkpointing |
| `worker_evaluator_v2.py` | Evaluation worker: polls `last.pt`, evaluates on synthetic + real data, promotes `best.pt` |
| `worker_generator.py` | Generates and serialises synthetic dataset batches as `.npz` files |
| `data_utils.py` | `FinetuneConfig`, `SyntheticDataGenerator`, real dataset loading with PFN filter and subsampling |

---

## 7  Running

### Phase 1 — encoder warm-up

```bash
sbatch step1_encoder.sbatch
```

### Phase 2 — full fine-tune

```bash
sbatch step2_finetune.sbatch
# Override defaults:  N_STEPS=500 LR=3e-5 sbatch step2_finetune.sbatch
```

### Final evaluation

```bash
sbatch eval_final.sbatch
# Or with a specific checkpoint:
CKPT=03_finetuning/checkpoints/phase1/best.pt sbatch eval_final.sbatch
```

### Local smoke test

```bash
cd 03_finetuning && python model.py
```

---

## References

- **TabPFN v2.5:** Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A.,
  Körfer, M., Hoo, B., Schirrmeister, R. T., & Hutter, F.  *Accurate
  predictions on small data with a tabular foundation model.*  Nature 637,
  319–326 (2025).  https://doi.org/10.1038/s41586-024-08328-6

- **PatchTST:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J.
  *A time series is worth 64 words: long-term forecasting with transformers.*
  ICLR 2023.  https://arxiv.org/abs/2211.14730

- **KernelSynth / Chronos:** Ansari, A. F., Stella, L., Turkmen, C., et al.
  *Chronos: Learning the language of time series.*  arXiv:2403.07815 (2024).

- **ROCKET:** Dempster, A., Petitjean, F., & Webb, G. I.  *ROCKET:
  Exceptionally fast and accurate time series classification using random
  convolutional kernels.*  Data Min Knowl Disc 34, 1454–1495 (2020).

- **HIVE-COTE 2.0:** Middlehurst, M., Large, J., Flynn, M. et al.
  *HIVE-COTE 2.0: a new meta ensemble for time series classification.*
  Mach Learn 110, 3211–3243 (2021).

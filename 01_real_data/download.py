#!/usr/bin/env python3
"""
Download 128 UCR and 30 UEA time series datasets via the aeon API.
Variable-length series are right-padded with NaN; missing values are kept as NaN.
Outputs a single CSV with shapes, flags, and PFN filter pass/fail.

See README.md in this folder for documentation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
DATA_UCR = HERE / "data" / "ucr"
DATA_UEA = HERE / "data" / "uea"

# ── Dataset lists (from aeon: univariate_equal_length + variable + missing; multivariate_equal_length + variable) ──
UCR_VARIABLE_LENGTH = {
    "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "PickupGestureWiimoteZ", "PLAID", "ShakeGestureWiimoteZ",
}
UCR_MISSING_VALUES = {
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "MelbournePedestrian",
}
UEA_VARIABLE_LENGTH = {
    "CharacterTrajectories", "InsectWingbeat", "JapaneseVowels", "SpokenArabicDigits",
}


def get_ucr_datasets() -> list[str]:
    """All 128 UCR univariate datasets."""
    from importlib import import_module
    for mod in ("aeon.datasets.tsc_datasets", "aeon.datasets.tsc_data_lists"):
        try:
            m = import_module(mod)
            return sorted(m.univariate)
        except Exception:
            pass
    return sorted([
        "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
        "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
        "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY",
        "CricketZ", "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame",
        "DodgerLoopWeekend", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
        "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour",
        "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain", "FreezerSmallTrain",
        "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1",
        "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty",
        "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
        "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
        "MedicalImages", "MelbournePedestrian", "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
        "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
        "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "PickupGestureWiimoteZ",
        "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "Plane", "PowerCons",
        "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
        "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
        "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
        "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
        "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2",
        "Trace", "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
        "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms",
        "WormsTwoClass", "Yoga",
    ])


def get_ucr_standard() -> list[str]:
    """112 UCR equal-length (standard bake-off set)."""
    from importlib import import_module
    for mod in ("aeon.datasets.tsc_datasets", "aeon.datasets.tsc_data_lists"):
        try:
            m = import_module(mod)
            return sorted(m.univariate_equal_length)
        except Exception:
            pass
    return [n for n in get_ucr_datasets()
            if n not in UCR_VARIABLE_LENGTH and n not in UCR_MISSING_VALUES and n != "Fungi"]


def get_uea_datasets() -> list[str]:
    """All 30 UEA multivariate datasets."""
    from importlib import import_module
    for mod in ("aeon.datasets.tsc_datasets", "aeon.datasets.tsc_data_lists"):
        try:
            m = import_module(mod)
            return sorted(m.multivariate)
        except Exception:
            pass
    return sorted([
        "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "CharacterTrajectories",
        "Cricket", "DuckDuckGeese", "EigenWorms", "Epilepsy", "EthanolConcentration", "ERing",
        "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
        "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery", "NATOPS", "PenDigits",
        "PEMS-SF", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
        "SpokenArabicDigits", "StandWalkJump", "UWaveGestureLibrary",
    ])


def get_uea_standard() -> list[str]:
    """26 UEA equal-length (standard bake-off set)."""
    from importlib import import_module
    for mod in ("aeon.datasets.tsc_datasets", "aeon.datasets.tsc_data_lists"):
        try:
            m = import_module(mod)
            return sorted(m.multivariate_equal_length)
        except Exception:
            pass
    return [n for n in get_uea_datasets() if n not in UEA_VARIABLE_LENGTH]


# ── Padding and conversion ────────────────────────────────────────────────────

def _pad_to_3d(X_list: list, univariate: bool) -> np.ndarray:
    """
    Convert list of variable-length series to (n, channels, max_t) with right-padding by NaN.
    Univariate: each element 1D or (1, t). Multivariate: each (m, t).
    """
    if not X_list:
        return np.array([]).reshape(0, 1, 0)
    if univariate:
        lengths = [np.asarray(x).flatten().shape[0] for x in X_list]
        max_t = max(lengths)
        n = len(X_list)
        out = np.full((n, 1, max_t), np.nan, dtype=np.float64)
        for i, x in enumerate(X_list):
            flat = np.asarray(x, dtype=np.float64).flatten()
            out[i, 0, : len(flat)] = flat
        return out
    else:
        first = np.asarray(X_list[0])
        if first.ndim == 1:
            first = first.reshape(1, -1)
        channels = first.shape[0]
        lengths = []
        for x in X_list:
            arr = np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            lengths.append(arr.shape[1])
        max_t = max(lengths)
        n = len(X_list)
        out = np.full((n, channels, max_t), np.nan, dtype=np.float64)
        for i, x in enumerate(X_list):
            arr = np.asarray(x, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            _, t = arr.shape
            out[i, :, :t] = arr
        return out


def _ensure_3d(X: np.ndarray) -> np.ndarray:
    """(n, t) -> (n, 1, t); ensure float for NaN compatibility."""
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    return np.asarray(X, dtype=np.float64)


# ── Download ──────────────────────────────────────────────────────────────────

def _read_shapes(name: str, data_dir: Path) -> dict | None:
    """Read shapes and flags from existing .npz (without loading full arrays)."""
    tr = data_dir / f"{name}_train.npz"
    te = data_dir / f"{name}_test.npz"
    if not (tr.exists() and te.exists()):
        return None
    try:
        with np.load(tr, allow_pickle=True) as d_tr, np.load(te, allow_pickle=True) as d_te:
            X_tr = d_tr["X"]
            n_train = int(X_tr.shape[0])
            n_channels = int(X_tr.shape[1])
            n_timesteps = int(X_tr.shape[2])
            n_test = int(d_te["X"].shape[0])
            n_classes = int(len(np.unique(np.concatenate([d_tr["y"], d_te["y"]]))))
            is_var = bool(d_tr.get("is_variable_length", np.array(False)).item())
            has_miss = bool(d_tr.get("has_missings", np.array(False)).item())
        return {
            "n_train": n_train,
            "n_channels": n_channels,
            "n_timesteps": n_timesteps,
            "n_test": n_test,
            "n_classes": n_classes,
            "is_variable_length": is_var,
            "has_missings": has_miss,
        }
    except Exception:
        return None


def _download_one(
    name: str,
    data_dir: Path,
    *,
    is_ucr: bool,
) -> dict | None:
    """
    Download one dataset via aeon. Variable-length → right-pad with NaN; missing values kept as NaN.
    Returns dict with n_train, n_channels, n_timesteps, n_test, n_classes, is_variable_length, has_missings.
    """
    cached = _read_shapes(name, data_dir)
    if cached:
        return cached

    from aeon.datasets import load_classification  # type: ignore
    try:
        X_tr, y_tr = load_classification(name, split="TRAIN")
        X_te, y_te = load_classification(name, split="TEST")
    except Exception as e:
        tqdm.write(f"  ERROR {name}: {e}")
        return None

    is_variable_length = name in (UCR_VARIABLE_LENGTH if is_ucr else UEA_VARIABLE_LENGTH)
    has_missings = name in UCR_MISSING_VALUES  # UEA has no missing-value set in aeon

    if isinstance(X_tr, list) or (hasattr(X_tr, "dtype") and X_tr.dtype == object):
        X_tr = _pad_to_3d(list(X_tr), univariate=is_ucr)
        X_te = _pad_to_3d(list(X_te), univariate=is_ucr)
        is_variable_length = True
    else:
        X_tr = _ensure_3d(np.asarray(X_tr))
        X_te = _ensure_3d(np.asarray(X_te))
        # Replace any sentinel missing value with NaN if present
        if np.isnan(X_tr).any() or np.isnan(X_te).any():
            has_missings = True
        # Known missing-value datasets: replace common sentinels with NaN
        if name in UCR_MISSING_VALUES:
            for arr in (X_tr, X_te):
                invalid = (arr == -1e10) | (arr == 1e10)
                if np.any(invalid):
                    arr[invalid] = np.nan
                    has_missings = True

    # Ensure labels are integer (aeon may return strings, e.g. InsectWingbeat)
    y_tr = np.asarray(y_tr).ravel()
    y_te = np.asarray(y_te).ravel()
    all_labels = np.concatenate([y_tr, y_te])
    if all_labels.dtype.kind in ("i", "u"):
        y_tr = y_tr.astype(np.int64)
        y_te = y_te.astype(np.int64)
    elif all_labels.dtype.kind == "f" and np.all(np.isfinite(all_labels)):
        y_tr = np.asarray(y_tr, dtype=np.int64)
        y_te = np.asarray(y_te, dtype=np.int64)
    else:
        # String or other: map to 0, 1, ... preserving order
        uniq = sorted(np.unique(all_labels), key=str)
        label_to_idx = {v: i for i, v in enumerate(uniq)}
        y_tr = np.array([label_to_idx[v] for v in y_tr], dtype=np.int64)
        y_te = np.array([label_to_idx[v] for v in y_te], dtype=np.int64)
    n_classes = int(len(np.unique(np.concatenate([y_tr, y_te]))))

    np.savez_compressed(
        data_dir / f"{name}_train.npz",
        X=X_tr,
        y=y_tr,
        is_variable_length=np.array(is_variable_length),
        has_missings=np.array(has_missings),
    )
    np.savez_compressed(
        data_dir / f"{name}_test.npz",
        X=X_te,
        y=y_te,
        is_variable_length=np.array(is_variable_length),
        has_missings=np.array(has_missings),
    )
    return {
        "n_train": int(X_tr.shape[0]),
        "n_channels": int(X_tr.shape[1]),
        "n_timesteps": int(X_tr.shape[2]),
        "n_test": int(X_te.shape[0]),
        "n_classes": n_classes,
        "is_variable_length": is_variable_length,
        "has_missings": has_missings,
    }


def download_datasets(
    names: list[str],
    data_dir: Path,
    label: str,
    is_ucr: bool,
) -> dict[str, dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, dict] = {}
    print(f"\nDownloading {len(names)} {label} datasets...")
    for name in tqdm(names, desc=label):
        info = _download_one(name, data_dir, is_ucr=is_ucr)
        if info:
            shapes[name] = info
    ok = len(shapes)
    failed = len(names) - ok
    print(f"  {ok}/{len(names)} OK" + (f", {failed} failed" if failed else ""))
    return shapes


# ── PFN filters: m*t <= 2000 and labels <= 10 ────────────────────────────────
# Datasets with n_train >= SUBSAMPLE_N will have subsample_train=True;
# loaders subsample train to exactly SUBSAMPLE_N with a fixed seed for
# reproducibility across all benchmarks and evaluations.

PFN_MT_MAX = 2_000
PFN_C_MAX = 10
SUBSAMPLE_N = 1_000


def pfn_passes(info: dict) -> bool:
    """True if dataset passes PFN filter (m*t<=2000, labels<=10)."""
    m = info.get("n_channels", 0)
    t = info.get("n_timesteps", 0)
    c = info.get("n_classes", 0)
    return (m * t) <= PFN_MT_MAX and c <= PFN_C_MAX


def needs_subsample(info: dict) -> bool:
    """True if n_train >= SUBSAMPLE_N; loader should subsample train to SUBSAMPLE_N."""
    return info.get("n_train", 0) >= SUBSAMPLE_N


# ── Summary CSV ────────────────────────────────────────────────────────────────

def build_summary_csv(
    ucr_shapes: dict[str, dict],
    uea_shapes: dict[str, dict],
    ucr_standard: list[str],
    uea_standard: list[str],
) -> pd.DataFrame:
    ucr_std_set = set(ucr_standard)
    uea_std_set = set(uea_standard)
    rows = []
    for name, info in sorted(ucr_shapes.items()):
        n_tr = info["n_train"]
        m = info["n_channels"]
        t = info["n_timesteps"]
        n_te = info["n_test"]
        rows.append({
            "dataset": name,
            "collection": "UCR",
            "train_shape": f"{n_tr}×{m}×{t}",
            "test_shape": f"{n_te}×{m}×{t}",
            "n_classes": info["n_classes"],
            "is_variable_length": info.get("is_variable_length", False),
            "has_missings": info.get("has_missings", False),
            "is_standard": name in ucr_std_set,
            "passes_pfn_filters": pfn_passes(info),
            "subsample_train": needs_subsample(info),
        })
    for name, info in sorted(uea_shapes.items()):
        n_tr = info["n_train"]
        m = info["n_channels"]
        t = info["n_timesteps"]
        n_te = info["n_test"]
        rows.append({
            "dataset": name,
            "collection": "UEA",
            "train_shape": f"{n_tr}×{m}×{t}",
            "test_shape": f"{n_te}×{m}×{t}",
            "n_classes": info["n_classes"],
            "is_variable_length": info.get("is_variable_length", False),
            "has_missings": info.get("has_missings", False),
            "is_standard": name in uea_std_set,
            "passes_pfn_filters": pfn_passes(info),
            "subsample_train": needs_subsample(info),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 128 UCR and 30 UEA datasets from aeon; output summary CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true", help="Only rebuild summary CSV from existing data")
    args = parser.parse_args()

    ucr_names = get_ucr_datasets()
    uea_names = get_uea_datasets()
    ucr_standard = get_ucr_standard()
    uea_standard = get_uea_standard()
    print(f"UCR: {len(ucr_names)} datasets (standard: {len(ucr_standard)})")
    print(f"UEA: {len(uea_names)} datasets (standard: {len(uea_standard)})")

    if not args.no_download:
        ucr_shapes = download_datasets(ucr_names, DATA_UCR, "UCR univariate", is_ucr=True)
        uea_shapes = download_datasets(uea_names, DATA_UEA, "UEA multivariate", is_ucr=False)
    else:
        ucr_shapes = {n: s for n in ucr_names if (s := _read_shapes(n, DATA_UCR)) is not None}
        uea_shapes = {n: s for n in uea_names if (s := _read_shapes(n, DATA_UEA)) is not None}
        print(f"\nLoaded shapes: UCR {len(ucr_shapes)}/{len(ucr_names)}, UEA {len(uea_shapes)}/{len(uea_names)}")

    df = build_summary_csv(ucr_shapes, uea_shapes, ucr_standard, uea_standard)
    out_path = HERE / "datasets_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")

    # Counts
    ucr_df = df[df["collection"] == "UCR"]
    uea_df = df[df["collection"] == "UEA"]
    ucr_pass = int(ucr_df["passes_pfn_filters"].sum())
    uea_pass = int(uea_df["passes_pfn_filters"].sum())
    ucr_std_pass = int((ucr_df["is_standard"] & ucr_df["passes_pfn_filters"]).sum())
    uea_std_pass = int((uea_df["is_standard"] & uea_df["passes_pfn_filters"]).sum())
    ucr_sub = int((ucr_df["passes_pfn_filters"] & ucr_df["subsample_train"]).sum())
    uea_sub = int((uea_df["passes_pfn_filters"] & uea_df["subsample_train"]).sum())

    print(f"\n--- PFN filters (m*T<=2000, labels<=10) ---")
    print(f"UCR: {ucr_pass}/{len(ucr_df)} pass PFN filters  ({ucr_sub} will be subsampled to {SUBSAMPLE_N} train)")
    print(f"UEA: {uea_pass}/{len(uea_df)} pass PFN filters  ({uea_sub} will be subsampled to {SUBSAMPLE_N} train)")
    print(f"UCR standard (112): {ucr_std_pass}/112 pass PFN filters")
    print(f"UEA standard (26): {uea_std_pass}/26 pass PFN filters")
    print("\nDone.")


if __name__ == "__main__":
    main()

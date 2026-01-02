# 3D Synthetic Dataset Generator with Temporal Dependencies

Generador de datasets sintéticos 3D con dependencias temporales para clasificación y regresión.

## Overview

Este módulo genera datasets con shape `(n_samples, n_features, t_timesteps)` donde:
- `n_samples`: Número de observaciones
- `n_features`: Número de features (nodos del grafo observados)
- `t_timesteps`: Longitud de la subsecuencia temporal

El proceso generador:
1. Construye un DAG causal (reutilizando componentes de 02_synthetic_generator_2D)
2. Propaga valores T veces con 3 tipos de inputs:
   - **Noise inputs**: Ruido aleatorio (como en 2D)
   - **Time inputs**: Funciones de `u = t/T` (tendencias, estacionalidad, decay)
   - **State inputs**: Memoria del timestep anterior (dependencias temporales)
3. Extrae subsecuencias de features y targets

## Input Types

### Noise Inputs
- Normal, Uniform, o Mixed
- Igual que en el generador 2D

### Time Inputs
Funciones de tiempo normalizado `u = t/T`:
- `constant`: 1
- `linear`: u
- `quadratic`: u²
- `cubic`: u³
- `tanh`: tanh(β(2u-1)), β ∈ LogUniform(0.5, 3.0)
- `sin_k`: sin(2πku), k ∈ {1,2,3,5}
- `cos_k`: cos(2πku), k ∈ {1,2,3,5}
- `exp_decay`: exp(-γu), γ ∈ LogUniform(0.5, 5.0)

### State Inputs
- Nodos designados como "estados" se guardan y usan como input en t+1
- Se normaliza: `tanh(α · s_{t-1})` donde α ∈ LogUniform(0.5, 2.0)
- En t=0, se inicializan con ruido

## Sampling Modes

### IID Mode
Cada sample es una secuencia independiente con ruido diferente.

### Sliding Window Mode
De una secuencia larga T, se extraen múltiples subsecuencias (pueden solaparse).

### Mixed Mode
Combinación de varias secuencias largas.

## Target Configuration

El target puede estar en diferentes posiciones temporales:
- **within**: Dentro de la subsecuencia de features (clasificación)
- **future_near**: 1-5 pasos adelante (predicción corto plazo)
- **future_far**: 6-20 pasos adelante (predicción largo plazo, menos frecuente)
- **past**: Antes de la subsecuencia (raro)

## Usage

```python
from generator import SyntheticDatasetGenerator3D, generate_3d_dataset

# Generación rápida
dataset = generate_3d_dataset(seed=42)
X, y = dataset.X, dataset.y  # (n, m, t), (n,)

# Con configuración custom
from config import PriorConfig3D
prior = PriorConfig3D(
    max_features=10,
    prob_classification=1.0,
    prob_sliding_window_mode=0.5
)
generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
dataset = generator.generate()

# Múltiples datasets
for i, dataset in enumerate(generator.generate_many(100)):
    print(f"Dataset {i}: {dataset.shape}")
```

## Module Structure

```
03_synthetic_generator_3D/
├── config.py              # PriorConfig3D, DatasetConfig3D
├── temporal_inputs.py     # Generadores de inputs (noise, time, state)
├── temporal_propagator.py # Propagación temporal por el DAG
├── sequence_sampler.py    # Extracción de subsecuencias (IID, sliding window, mixed)
├── feature_selector.py    # Selección de features y target
├── generator.py           # Clase principal SyntheticDatasetGenerator3D
├── sanity_checks.py       # Validación de datasets generados
└── README.md
```

## Constraints

- Max 10,000 samples (train + test)
- Max 15 features
- Max 1,000 timesteps
- Max 10 classes

## Sanity Checks

```bash
cd 03_synthetic_generator_3D
python sanity_checks.py --n_datasets 50 --seed 42
```

Los sanity checks verifican:
- Shapes correctos
- Datasets aprendibles (lift > 0)
- Variedad en modos de sampling
- Variedad en tipos de target offset
- Sin errores de generación

## Key Differences from 2D Generator

| Aspecto | 2D | 3D |
|---------|----|----|
| Shape | (n, m) | (n, m, t) |
| Inputs | Solo ruido | Ruido + Tiempo + Estado |
| Dependencias | Ninguna | Temporal (memoria) |
| Target | Un nodo | Un nodo en un timestep |
| Sampling | Una propagación | T propagaciones + extracción |


# 3D Synthetic Dataset Generator with Temporal Dependencies

Generador de datasets sintéticos 3D con dependencias temporales para clasificación de series temporales.

## Overview

Este módulo genera datasets con shape `(n_samples, n_features, t_timesteps)` donde:
- `n_samples`: Número de observaciones
- `n_features`: Número de features (nodos del grafo observados)
- `t_timesteps`: Longitud de la subsecuencia temporal

## Arquitectura

```
                    ┌──────────────────────────────────────────┐
                    │           DAG Causal                     │
                    │  (construido con orden topológico)       │
                    └──────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              ┌─────┴─────┐                      ┌──────┴──────┐
              │  t = 0    │        ...           │  t = T-1    │
              └─────┬─────┘                      └──────┬──────┘
                    │                                   │
    Inputs:  Noise + Time(0) + State(init)    Noise + Time(T-1) + State(t-1)
                    │                                   │
                    ▼                                   ▼
              Propagation                         Propagation
                    │                                   │
                    └───────────────┬───────────────────┘
                                    │
                              Extract Windows
                                    │
                                    ▼
                              (n, m, t), y
```

## Input Types (Nodos Raíz)

### Noise Inputs
- Normal N(0, σ²) o Uniform U(-a, a)
- Valores nuevos en cada timestep
- Proveen variabilidad entre samples

### Time Inputs
Funciones determinísticas de tiempo normalizado `u = t/T`:
- `linear`: u
- `quadratic`: u²
- `cubic`: u³
- `tanh`: tanh(β(2u-1)), β ∈ LogUniform(0.5, 3.0)
- `sin_k`: sin(2πku), k ∈ {1,2,3,5}
- `cos_k`: cos(2πku), k ∈ {1,2,3,5}
- `exp_decay`: exp(-γu), γ ∈ LogUniform(0.5, 5.0)
- `log`: log(u + 0.1)

### State Inputs
- Memoria del timestep anterior
- En t=0 se inicializan con ruido
- Normalizados: `tanh(α · s_{t-1})`
- Permiten dependencias temporales (AR-like)

## Transformaciones

Cada nodo no-raíz tiene **una** transformación (misma que 2D):

| Tipo | Descripción |
|------|-------------|
| **NN** | weights × padres + bias → activación → ruido |
| **Tree** | Decision tree sobre subset de padres |
| **Discretization** | Distancia a prototipos → categoría normalizada |

### Activaciones Disponibles (12)
```python
['identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 
 'rank', 'square', 'power', 'softplus', 'step', 'mod']
```

## Sampling Modes

### IID Mode
```
Sequence 1: ────────────────────────────
Sequence 2: ────────────────────────────
    ...
Sequence N: ────────────────────────────
```
Cada sample es una secuencia independiente con ruido diferente.

### Sliding Window Mode
```
Long sequence: ══════════════════════════════════════════
Windows:       [───────]
                 [───────]
                   [───────]
                     [───────]
```
De una secuencia larga T, se extraen múltiples ventanas (pueden solaparse).

### Mixed Mode
```
Seq 1: ══════════════════════
        [───] [───] [───]
Seq 2: ══════════════════════
        [───] [───] [───]
```
Varias secuencias largas, múltiples ventanas por secuencia.

## Target Configuration

El target puede estar en diferentes posiciones:
- **within**: Dentro de la ventana de features
- **future_near**: 1-5 pasos después de la ventana
- **future_far**: 6-20 pasos adelante
- **past**: Antes de la ventana (raro)

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
    prob_sliding_window_mode=0.6,
    max_complexity=5_000_000  # Limitar complejidad
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
├── dag_utils.py           # DAG wrapper sobre 2D (usa orden topológico)
├── temporal_inputs.py     # Generadores de inputs (noise, time, state)
├── temporal_propagator.py # Propagación temporal optimizada
├── sequence_sampler.py    # Extracción de subsecuencias
├── feature_selector.py    # Selección de features y target
├── generator.py           # Clase principal
├── sanity_checks.py       # Validación completa + comparación con reales
├── discriminator_analysis.py  # Análisis sintético vs real
├── visualize_dag.py       # Visualización de grafos
└── README.md
```

## Parámetros Clave

### Límites de Tamaño
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| max_samples | 10,000 | Samples máximos |
| max_features | 15 | Features máximos |
| max_t_subseq | 1,000 | Timesteps máximos por ventana |
| max_T_total | 5,000 | Timesteps totales máximos |
| max_classes | 10 | Clases máximas |
| max_complexity | 10,000,000 | n_samples × T_total × n_nodes |

### Estructura del Grafo
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| n_nodes_range | (12, 300) | Nodos del DAG |
| density_range | (0.01, 0.8) | Densidad de edges |
| n_roots_range | (3, 40) | Número de roots |
| max_roots_fraction | 0.25 | Roots ≤ 25% de nodos |

### Distribución de Inputs
| Tipo | Mínimo | Descripción |
|------|--------|-------------|
| Noise | 1 | Variabilidad entre samples |
| Time | 1 | Tendencias temporales |
| State | 1 | Dependencias temporales |

### Probabilidades de Modo
| Modo | Probabilidad |
|------|--------------|
| IID | 20% |
| Sliding Window | 60% |
| Mixed | 20% |

## Optimizaciones de Rendimiento

El generador incluye varias optimizaciones:

1. **Límite de complejidad**: Si `n_samples × T_total × n_nodes > max_complexity`, reduce parámetros automáticamente

2. **Propagación vectorizada**: Arrays pre-asignados en lugar de diccionarios

3. **Cache de timeseries**: Los timeseries se cachean para extracción eficiente

4. **Batch processing**: Múltiples samples se procesan en paralelo

Tiempo típico: **~0.8s por dataset** (promedio)

## Sanity Checks

```bash
cd 03_synthetic_generator_3D
python sanity_checks.py
```

Los sanity checks incluyen:

1. **Basic Stats**: Shapes, modos, NaN rates
2. **Learnability**: Models beat baseline
3. **Temporal Characteristics**: Autocorrelación, tendencias
4. **Mode Comparison**: IID vs Sliding vs Mixed
5. **Label Permutation**: Sin data leakage
6. **Comparison with Real**: Distribuciones vs UCR/UEA datasets
7. **Difficulty Spectrum**: Variedad de dificultades
8. **Input Type Distribution**: Balance noise/time/state

## Comparación con Datasets Reales

El check 6 compara con datasets reales del PKL:
- n_samples, t_length: distribuciones similares
- Autocorrelación: sintéticos tienen menos AC(1) que reales
- Varianza: reales tienen más variabilidad

## Diferencias con Generador 2D

| Aspecto | 2D | 3D |
|---------|----|----|
| Shape | (n, m) | (n, m, t) |
| Inputs | Solo ruido | Ruido + Tiempo + Estado |
| Dependencias | Ninguna | Temporal (memoria) |
| Target | Un nodo | Un nodo en un timestep |
| Sampling | Una propagación | T propagaciones + extracción |
| Complejidad | O(n × nodes) | O(n × T × nodes) |

## Discriminator Analysis

Análisis de distinguibilidad sintético vs real:

```bash
python discriminator_analysis.py
```

Genera:
- Features de datasets (34 métricas)
- Clasificador Random Forest para distinguir
- Feature importance
- Visualizaciones por dataset

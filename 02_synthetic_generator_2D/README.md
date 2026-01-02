# Synthetic Dataset Generator for TabPFN

Implementación del generador de datos sintéticos tabulares basado en el paper de TabPFN. Genera datasets diversos para entrenar modelos que aprenden a resolver problemas tabulares en general.

## Arquitectura

```
Root Nodes (Noise) → DAG Transformations → Feature Selection → Post-processing → (X, y)
     ↓                      ↓                     ↓                  ↓
  Normal/Uniform    NN, Tree, Discretize    Relevant/Irrelevant   Warp, Missing
```

## Quick Start

```python
from generator import SyntheticDatasetGenerator

# Generar un dataset
gen = SyntheticDatasetGenerator(seed=42)
dataset = gen.generate()

X, y = dataset.X, dataset.y
print(f"Shape: {X.shape}, Classes: {dataset.n_classes}")
```

## Módulos

| Archivo | Función |
|---------|---------|
| `config.py` | `PriorConfig` (distribuciones) y `DatasetConfig` (instancia) |
| `dag_builder.py` | Construcción del DAG con redirection sampling |
| `transformations.py` | Transformaciones: NN, Tree, Discretization, Identity |
| `row_generator.py` | Propagación de valores por el grafo |
| `feature_selector.py` | Selección de features y target |
| `post_processing.py` | Warping (Kumaraswamy), cuantización, missing values |
| `generator.py` | Clase principal |
| `sanity_checks.py` | Validación de datasets generados |
| `analyze_value_ranges.py` | Análisis de rangos de valores |

## Parámetros Clave (según paper)

### Tamaño
- **n_rows**: Uniforme en [50, 2048]
- **n_features**: Beta(0.95, 8.0) escalado a [1, 160]
- **max_cells**: 75,000 (n_rows × n_features)

### Estructura del Grafo
- **n_nodes**: Log-uniform en [50, 600] (nodos latentes)
- **redirection_prob**: Gamma(2.0, 5.0) - controla densidad del grafo

### Transformaciones
```
NN:            ~50% (linear + activación)
Tree:          ~20% (decision tree piecewise)
Discretization: ~15% (categorización por prototipos)
Identity:      ~15% (suma ponderada + ruido)
```

### Activaciones (NN)
`relu, tanh, sigmoid, leaky_relu, elu, softplus, power`

### Ruido en Raíces
- **Normal**: N(0, σ²) con σ ~ Uniform[0.1, 2.0]
- **Uniform**: U(-a, a) con a ~ Uniform[0.5, 2.0]
- **Mixed**: Selección aleatoria entre Normal y Uniform por nodo

### Número de Clases
- Gamma(2.0, 2.0) + offset de 2
- Limitado a min(10, n_rows/10) para balance

## Rangos de Valores Típicos

| Etapa | Rango p1-p99 | Mean | Std |
|-------|--------------|------|-----|
| Root nodes | [-0.6, 0.5] | ~0 | 0.3 |
| Depth 4 | [-1.4, 7.4] | 0.3 | 1.1 |
| Features (X) | [-4.5, 1.7] | 0.2 | 1.0 |

## Sanity Checks

```bash
python sanity_checks.py --n 100 --seed 42
```

Verifica:
- ✅ Distribución de accuracy (no trivial ni imposible)
- ✅ Variabilidad de rankings entre modelos
- ✅ Features relevantes vs irrelevantes
- ✅ Label permutation test (sin data leakage)
- ✅ Learning curves (mejora con más datos)
- ✅ Invariancia a permutaciones

## Análisis de Rangos

```bash
python analyze_value_ranges.py --seed 42
```

Muestra valores en cada profundidad del grafo y por tipo de transformación.

## Prevención de Data Leakage

El generador excluye de los features:
1. El nodo target
2. Los **padres directos** del target (evita que X contenga la pre-transformación de y)

## Configuración Personalizada

```python
from config import PriorConfig

prior = PriorConfig(
    n_rows_range=(100, 500),
    n_nodes_range=(30, 100),
    prob_classification=1.0,  # Solo clasificación
    prob_missing_values=0.0,  # Sin missing values
)

gen = SyntheticDatasetGenerator(prior=prior, seed=42)
```

## Tests

```bash
python tests.py
```

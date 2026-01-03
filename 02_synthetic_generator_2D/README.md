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
| `dag_builder.py` | Construcción del DAG con orden topológico y densidad controlable |
| `transformations.py` | Transformaciones: NN (con 12 activaciones), Tree, Discretization |
| `row_generator.py` | Propagación de valores por el grafo |
| `feature_selector.py` | Selección de features y target |
| `post_processing.py` | Warping (Kumaraswamy), cuantización, missing values |
| `generator.py` | Clase principal |
| `sanity_checks.py` | Validación completa de datasets generados |
| `tests.py` | Tests unitarios |

## Parámetros Clave

### Tamaño
- **n_samples**: Uniforme en [50, 2048]
- **n_features**: Beta(0.95, 8.0) escalado a [1, 160]
- **max_cells**: 75,000 (n_samples × n_features)

### Estructura del Grafo
- **n_nodes**: Log-uniform en [50, 600] (nodos latentes)
- **density**: Uniforme en [0.01, 0.8] - controla cuántos edges adicionales se agregan
- **n_roots_range**: (3, 15) - número de nodos raíz (inputs)

### Transformaciones
Cada nodo no-raíz tiene exactamente **una** transformación que toma todos sus padres como input:

| Tipo | Prob | Descripción |
|------|------|-------------|
| NN | ~60% | Linear combination + activación + ruido |
| Tree | ~25% | Decision tree con subset de features de padres |
| Discretization | ~15% | Distancia a prototipos → categoría normalizada |

### Activaciones (12 funciones del paper)
```python
['identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 
 'rank', 'square', 'power', 'softplus', 'step', 'mod']
```

- **identity**: f(x) = x (transformación lineal)
- **log**: f(x) = log(|x| + 1)
- **sigmoid**: f(x) = 1 / (1 + e^(-x))
- **abs**: f(x) = |x|
- **sin**: f(x) = sin(x)
- **tanh**: f(x) = tanh(x)
- **rank**: f(x) = percentil rank
- **square**: f(x) = x²
- **power**: f(x) = |x|^α, α ∈ [0.5, 3]
- **softplus**: f(x) = log(1 + e^x)
- **step**: f(x) = 1 si x > 0, else 0
- **mod**: f(x) = x mod m, m ∈ [0.5, 2]

### Discretización
- Recibe vector de padres
- Calcula distancia a K prototipos (K ∈ [2, 8])
- Asigna categoría del prototipo más cercano
- Normaliza: output = categoría / K (para usar en grafo)
- Agrega ruido gaussiano

### Decision Tree
- Selecciona subset de features de los padres (tree_max_features_fraction=0.7)
- Genera árbol con profundidad [2, 5]
- Cada nodo: (feature_idx, threshold, left_val, right_val)

### Ruido en Transformaciones
Todas las transformaciones agregan ruido gaussiano N(0, σ²) al final.

### Número de Clases
- Gamma(2.0, 2.0) + offset de 2
- Limitado a min(10, n_samples/min_samples_per_class)
- **min_samples_per_class**: 10 (mínimo de samples por clase)

## Construcción del DAG

El DAG se construye usando **orden topológico**:

1. Asignar orden aleatorio a todos los nodos
2. Determinar número de roots (3-15)
3. Calcular edges objetivo basado en `density`
4. Agregar edges solo de nodos con orden menor a mayor (garantiza aciclicidad)
5. Asegurar conectividad (cada no-root tiene al menos 1 padre)

Esto permite controlar la **densidad** del grafo:
- density=0: grafo mínimo (árbol)
- density=1: DAG máximamente denso

## Subgrafos Desconectados

El generador puede crear subgrafos desconectados para features irrelevantes:
- Probabilidad: 30%
- Cada subgrafo tiene mínimo 3 nodos
- Subgrafos pueden tener múltiples roots
- El subgrafo principal retiene al menos 60% de los nodos

## Sanity Checks

```bash
python sanity_checks.py
```

Verifica:
- ✅ Distribución de accuracy (no trivial ni imposible)
- ✅ Variabilidad de rankings entre modelos
- ✅ Features relevantes vs irrelevantes
- ✅ Label permutation test (sin data leakage)
- ✅ Learning curves (mejora con más datos)
- ✅ Invariancia a permutaciones
- ✅ Discretización correcta (prototipos, categorías, entropía)
- ✅ Visualización de DAGs generados

## Selección de Features

El generador excluye de los features solo el **nodo target** (lo que queremos predecir).

Los padres del target **sí pueden ser features** - son información valiosa para la predicción. No hay data leakage porque los features son valores de entrada y el target es el resultado de la transformación.

## Configuración Personalizada

```python
from config import PriorConfig

prior = PriorConfig(
    n_samples_range=(100, 500),
    n_nodes_range=(30, 100),
    density_range=(0.1, 0.5),
    prob_classification=1.0,
    prob_missing_values=0.0,
    min_samples_per_class=20,
    activations=['identity', 'tanh', 'sigmoid', 'relu']
)

gen = SyntheticDatasetGenerator(prior=prior, seed=42)
```

## Tests

```bash
python tests.py
```

Incluye tests para:
- Construcción de DAG
- Todas las transformaciones
- Todas las activaciones
- Densidad del grafo
- Múltiples roots

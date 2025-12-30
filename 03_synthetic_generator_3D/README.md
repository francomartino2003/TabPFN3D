# 03 - Synthetic Generator 3D (Time Series Classification)

Este módulo genera datasets sintéticos de clasificación de series temporales, extendiendo la metodología de TabPFN a datos 3D.

## Formato de Datos

- **Input X**: `(n_samples, n_features, n_timesteps)` - Series temporales multivariadas
- **Output y**: `(n_samples,)` - Etiquetas de clasificación

## Proceso de Generación

### 1. Crear DAG Base
Similar al generador 2D, se crea un grafo causal dirigido acíclico (DAG) que define la estructura espacial.

### 2. Desenrollar Temporalmente
El DAG base se copia T veces (una por timestep).

### 3. Conexiones Temporales
Se agregan conexiones entre timesteps (siempre hacia el futuro para mantener aciclicidad):

#### Tipos de Conexiones

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **self** | Mismo nodo a través del tiempo | `node_i(t) → node_i(t+k)` |
| **cross** | Nodo diferente a través del tiempo | `node_i(t) → node_j(t+k)` |
| **many_to_one** | Múltiples nodos → un hub | Hub temporal |
| **one_to_many** | Un nodo → múltiples targets | Broadcast espacial |
| **broadcast_multiskip** | Mismo nodo a múltiples skips con decay | AR(p)-like: `node(t) → node(t+1), node(t+2)...` con pesos decrecientes |
| **conditional_lag** | Skip depende del valor | Lag switching / cambio de régimen |
| **conditional_dest** | Destino depende del valor | State-dependent dynamics |

#### Características Avanzadas

- **Skips**: Distribución geométrica (skips pequeños más probables)
- **Multi-skip con decay**: Simula dependencias AR(p) con peso = decay^skip
- **Conexiones parciales**: Algunas solo activas para subsecuencias de T
- **Conexiones condicionales**: El skip o destino depende del valor del nodo (tipo árbol)

### 4. Propagación
**CLAVE**: Cada subgrafo t recibe **ruido independiente** en sus nodos raíz (igual que en 2D).
Las dependencias temporales surgen únicamente de las conexiones entre grafos.

- Los nodos raíz de cada timestep obtienen ruido fresco
- El ruido puede ser AR(1) correlacionado si está configurado
- Las transformaciones temporales aplican pesos (para multi-skip con decay)

### 5. Selección de Features y Target
- **Features**: Nodos seleccionados del grafo base, observados en una ventana temporal continua
- **Target**: Un nodo en un timestep específico (puede estar antes, dentro, o después de la ventana de features)

### 6. Post-processing
- Warping temporal
- Cuantización
- Valores faltantes

## Uso Rápido

```python
from generator import SyntheticDatasetGenerator3D
from config import PriorConfig3D

# Generador con prior por defecto
generator = SyntheticDatasetGenerator3D(seed=42)

# Generar dataset
dataset = generator.generate()

print(f"X shape: {dataset.X.shape}")  # (n_samples, n_features, n_timesteps)
print(f"y shape: {dataset.y.shape}")  # (n_samples,)
print(f"Classes: {dataset.n_classes}")
```

## Configuración Personalizada

```python
# Prior para series univariadas cortas (común en UCR/UEA)
prior = PriorConfig3D(
    n_samples_range=(100, 500),
    n_features_range=(1, 3),
    prob_univariate=0.7,      # 70% chance de 1 sola feature
    n_timesteps_range=(50, 200),
    max_classes=5,
)

generator = SyntheticDatasetGenerator3D(prior=prior)
```

## Hiperparámetros Clave

### Tamaño
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `n_samples_range` | (50, 10000) | Número de observaciones |
| `n_features_range` | (1, 15) | Número de features/canales |
| `n_timesteps_range` | (20, 1000) | Longitud temporal |
| `prob_univariate` | 0.4 | Prob. de dataset univariado |

### Conexiones Temporales
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `prob_temporal_self` | 0.25 | Prob. self-connection |
| `prob_temporal_cross` | 0.20 | Prob. cross-connection |
| `prob_temporal_many_to_one` | 0.15 | Prob. hub temporal |
| `prob_temporal_one_to_many` | 0.15 | Prob. broadcast |
| `prob_temporal_broadcast_multiskip` | 0.10 | Prob. AR-like multi-skip |
| `prob_temporal_conditional_lag` | 0.08 | Prob. lag switching |
| `prob_temporal_conditional_dest` | 0.07 | Prob. destination switching |
| `temporal_skip_geometric_p` | 0.3 | P(skip=1), geométrico |
| `n_temporal_patterns_range` | (3, 15) | # patrones temporales |
| `prob_partial_time_range` | 0.2 | Prob. conexión parcial |

### Posición del Target
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `prob_target_before` | 0.1 | Target antes de features (raro) |
| `prob_target_within` | 0.4 | Target durante features |
| `prob_target_after` | 0.5 | Target después (predicción) |

### Ruido Temporal
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `prob_correlated_noise` | 0.3 | Prob. de ruido AR(1) |
| `noise_ar_coef_range` | (0.1, 0.9) | Coeficiente AR |

## Compatibilidad con 2D

Para usar con modelos que esperan entrada 2D:

```python
X_2d, y = dataset.to_2d()
# X_2d shape: (n_samples, n_features * n_timesteps)
```

## Ejecutar Tests

```bash
cd 03_synthetic_generator_3D
python tests.py
```

## Ejecutar Demo

```bash
python demo.py
```

## Estructura de Archivos

```
03_synthetic_generator_3D/
├── config.py               # PriorConfig3D, DatasetConfig3D, TemporalConnectionConfig
├── temporal_connections.py # Tipos ricos de conexiones temporales
├── temporal_dag_builder.py # Construcción del DAG desenrollado
├── row_generator_3d.py     # Propagación a través del tiempo
├── feature_selector_3d.py  # Selección de features y target
├── generator.py            # Clase principal
├── demo.py                 # Demostración
├── tests.py                # Tests unitarios
└── README.md
```

## Comparación con Generador 2D

| Aspecto | 2D | 3D |
|---------|----|----|
| Input shape | (n, m) | (n, m, t) |
| DAG | Simple | Desenrollado T veces |
| Conexiones | Solo espaciales | Espaciales + Temporales |
| Target | Un nodo | Un nodo en un timestep |
| Ruido | i.i.d. | Puede ser AR(1) |

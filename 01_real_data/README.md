# TabPFN 3D - Real Data Analysis (AEON)

Este directorio contiene el código y datos para analizar series temporales reales (no sintéticas) que se usarán para validar el modelo final. Los datos provienen del archivo AEON (UCR/UEA Time Series Classification Archive).

## Estructura

```
01_real_data/
├── src/                          # Código fuente
│   ├── data_loader.py            # Clase TimeSeriesDataset para encapsular datos
│   ├── load_classification_datasets.py  # Carga datasets de clasificación UCR/UEA
│   ├── load_forecasting_datasets.py    # Carga datasets de forecasting Monash
│   ├── time_series_statistics.py        # Cálculo de estadísticas
│   └── analyze_all_datasets.py         # Script principal de análisis
├── AEON/                          # Datos de AEON (UCR/UEA Archive)
│   ├── benchmarks/               # Benchmarks de bakeoff 2017
│   │   ├── accuracy/
│   │   ├── auroc/
│   │   ├── balacc/
│   │   ├── f1/
│   │   ├── fittime/
│   │   ├── logloss/
│   │   ├── memoryusage/
│   │   └── predicttime/
│   └── data/                      # Datasets y estadísticas
│       ├── classification_datasets.pkl
│       ├── classification_stats.csv
│       ├── classification_stats.json
│       ├── classification_distributions.png
│       └── classification_relationships.png
└── notebooks/
    └── 01_statistical_analysis.ipynb  # Análisis interactivo
```

## Uso

### 1. Descargar y analizar datasets de clasificación

```bash
python src/analyze_all_datasets.py
```

Esto:
- Descarga todos los datasets de clasificación UCR/UEA usando `aeon`
- Guarda train y test por separado en `AEON/data/classification_datasets.pkl`
- Calcula estadísticas básicas (shape, length, dimensions, classes, etc.)
- Guarda resultados en CSV y JSON
- Genera visualizaciones

### 2. Análisis interactivo

Abrir `notebooks/01_statistical_analysis.ipynb` para análisis interactivo.

## Datasets

### Clasificación (UCR/UEA Archive)
- Fuente: https://www.timeseriesclassification.com/dataset.php
- Total: ~181 datasets
- Cargados usando `aeon.datasets.load_classification`
- Train y test se guardan por separado

### Forecasting (Monash Archive)
- Fuente: https://forecastingdata.org/
- Pendiente de implementación

## Estadísticas calculadas

Para cada dataset se calcula:
- `n_samples`: Número total de muestras (train + test)
- `length`: Longitud temporal (timesteps)
- `n_dimensions`: Número de dimensiones/variables
- `n_classes`: Número de clases (para clasificación)
- `train_size`: Tamaño del conjunto de entrenamiento
- `test_size`: Tamaño del conjunto de prueba
- `missing_pct`: Porcentaje de valores faltantes
- `has_variable_length`: Si tiene series de longitud variable (padded)

## Benchmarks

Los benchmarks de bakeoff 2017 se guardan en `AEON/benchmarks/` con métricas:
- accuracy
- auroc
- balacc
- f1
- logloss
- fittime
- predicttime
- memoryusage

"""
Script de prueba para verificar que la configuración funciona correctamente
"""
import sys
from pathlib import Path

print("=" * 80)
print("PRUEBA DE CONFIGURACIÓN - TabPFN 3D")
print("=" * 80)

# Verificar imports
print("\n1. Verificando imports...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller
    print("   ✓ Imports básicos OK")
except ImportError as e:
    print(f"   ✗ Error en imports básicos: {e}")
    sys.exit(1)

try:
    from aeon.datasets import load_classification, get_dataset_names
    print("   ✓ aeon instalado correctamente")
except ImportError as e:
    print(f"   ✗ aeon no instalado: {e}")
    print("   Instalar con: pip install aeon")
    sys.exit(1)

# Verificar módulos del proyecto
print("\n2. Verificando módulos del proyecto...")
try:
    from src.data_loader import TimeSeriesDataset
    from src.time_series_statistics import TimeSeriesStatistics
    from src.load_classification_datasets import (
        get_all_classification_datasets,
        load_single_classification_dataset
    )
    print("   ✓ Módulos del proyecto OK")
except ImportError as e:
    print(f"   ✗ Error en módulos del proyecto: {e}")
    sys.exit(1)

# Verificar estructura de carpetas
print("\n3. Verificando estructura de carpetas...")
folders = [
    "data/real/classification",
    "data/real/forecasting",
    "data/real/metadata",
    "src",
    "notebooks"
]

all_ok = True
for folder in folders:
    path = Path(folder)
    if path.exists():
        print(f"   ✓ {folder}")
    else:
        print(f"   ✗ {folder} - NO EXISTE")
        all_ok = False

if not all_ok:
    print("\n   Creando carpetas faltantes...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print("   ✓ Carpetas creadas")

# Probar carga de un dataset
print("\n4. Probando carga de dataset de ejemplo...")
try:
    dataset = load_single_classification_dataset("GunPoint", verbose=True)
    if dataset:
        print(f"   ✓ Dataset cargado: {dataset}")
        print(f"   ✓ Shape: {dataset.X.shape}")
        print(f"   ✓ Info: {dataset.get_info()}")
        
        # Probar cálculo de estadísticas
        print("\n5. Probando cálculo de estadísticas...")
        stats_dict = TimeSeriesStatistics.compute_dataset_stats(dataset)
        print(f"   ✓ Estadísticas calculadas")
        print(f"   ✓ Número de muestras: {stats_dict['n_samples']}")
        print(f"   ✓ Longitud temporal: {stats_dict['n_timesteps']}")
        print(f"   ✓ Número de canales: {stats_dict['n_channels']}")
    else:
        print("   ✗ No se pudo cargar el dataset")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Verificar lista de datasets disponibles
print("\n6. Verificando lista de datasets disponibles...")
try:
    dataset_names = get_all_classification_datasets()
    print(f"   ✓ {len(dataset_names)} datasets disponibles")
    print(f"   ✓ Primeros 5: {dataset_names[:5]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("CONFIGURACIÓN COMPLETA")
print("=" * 80)
print("\nPróximos pasos:")
print("1. Ejecutar: python src/analyze_all_datasets.py")
print("2. O abrir: notebooks/01_statistical_analysis.ipynb")
print("\n")


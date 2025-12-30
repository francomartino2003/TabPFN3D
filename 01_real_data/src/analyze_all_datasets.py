"""
Script principal para analizar todos los datasets
"""
import sys
from pathlib import Path

# Agregar el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import pickle
from typing import List, Dict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from src.data_loader import TimeSeriesDataset
from src.time_series_statistics import TimeSeriesStatistics
from src.load_classification_datasets import load_all_classification_datasets



def analyze_classification_datasets(output_dir: Path, 
                                   max_datasets: int = None,
                                   save_datasets: bool = True) -> pd.DataFrame:
    """
    Analiza todos los datasets de clasificación
    
    Args:
        output_dir: Directorio para guardar resultados
        max_datasets: Número máximo de datasets a analizar
        save_datasets: Si True, guarda los datasets cargados
        
    Returns:
        DataFrame con estadísticas agregadas
    """
    print("=" * 80)
    print("ANÁLISIS DE DATASETS DE CLASIFICACIÓN")
    print("=" * 80)
    
    # Asegurar que output_dir es relativo al directorio raíz del proyecto
    if not output_dir.is_absolute():
        # Si estamos en src/, subir un nivel
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datasets
    datasets_path = output_dir / "classification_datasets.pkl"
    if datasets_path.exists() and save_datasets:
        print(f"\nCargando datasets desde {datasets_path}...")
        with open(datasets_path, 'rb') as f:
            datasets = pickle.load(f)
        print(f"Cargados {len(datasets)} datasets")
    else:
        print("\nDescargando datasets de clasificación...")
        datasets = load_all_classification_datasets(
            max_datasets=max_datasets,
            save_path=datasets_path if save_datasets else None,
            verbose=True
        )
    
    if not datasets:
        print("No se pudieron cargar datasets de clasificación")
        return pd.DataFrame()
    
    # Calcular estadísticas básicas (ligeras)
    print("\nCalculando estadísticas básicas...")
    all_stats = []
    simple_stats = []
    
    for dataset in tqdm(datasets, desc="Computing stats"):
        stats = TimeSeriesStatistics.compute_dataset_stats_simple(dataset)
        simple_stats.append(stats)
        all_stats.append(dataset.get_info())
    
    # Verificar si tienen benchmarks (usando los benchmarks descargados)
    print("\nVerificando benchmarks...")
    base_dir = Path(__file__).parent.parent
    benchmarks_dir = base_dir / "AEON" / "benchmarks"
    
    # Obtener lista de datasets con benchmarks desde los archivos CSV
    # Estructura: cada CSV tiene primera columna con nombre del dataset (sin header)
    # Las demás columnas son diferentes runs del modelo
    datasets_with_benchmarks = set()
    if benchmarks_dir.exists():
        # Buscar en cualquier carpeta de métricas
        for metric_dir in benchmarks_dir.iterdir():
            if metric_dir.is_dir():
                for csv_file in metric_dir.glob("*.csv"):
                    try:
                        # Leer CSV: primera columna es "Resamples," y luego los nombres de datasets están en la primera columna de cada fila
                        # Estructura: Resamples,0,1,2,... (header)
                        #           DatasetName,value1,value2,... (datos)
                        df_bench = pd.read_csv(csv_file)
                        
                        if len(df_bench) > 0:
                            # La primera columna contiene los nombres de los datasets
                            first_col_name = df_bench.columns[0]
                            # Obtener nombres de datasets desde la primera columna
                            dataset_names = df_bench[first_col_name].astype(str).tolist()
                            
                            # Normalizar nombres (lowercase, sin espacios, sin guiones bajos, sin guiones)
                            normalized_names = [str(name).lower().replace('_', '').replace(' ', '').replace('-', '') 
                                              for name in dataset_names 
                                              if pd.notna(name) and str(name).strip() and str(name).lower() != 'resamples']
                            datasets_with_benchmarks.update(normalized_names)
                    except Exception as e:
                        continue
    
    # Agregar información de benchmarks a las estadísticas
    for stats in simple_stats:
        dataset_name_normalized = stats['name'].lower().replace('_', '').replace(' ', '')
        stats['has_benchmark'] = dataset_name_normalized in datasets_with_benchmarks
    
    # Crear DataFrame para resumen (solo para imprimir)
    df_stats = pd.DataFrame(simple_stats)
    
    # Guardar estadísticas en JSON
    stats_path = output_dir / "classification_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(simple_stats, f, indent=2, default=str)
    print(f"Estadísticas guardadas en {stats_path}")
    
    # Imprimir resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE DATASETS DE CLASIFICACIÓN")
    print("=" * 80)
    print(f"\nTotal de datasets: {len(datasets)}")
    print(f"\nDistribución de shapes:")
    if 'length' in df_stats.columns and 'n_dimensions' in df_stats.columns:
        print(df_stats[['n_samples', 'length', 'n_dimensions']].describe())
    else:
        print(df_stats[['n_samples', 'n_timesteps', 'n_channels']].describe())
    print(f"\nValores faltantes:")
    print(f"  Promedio: {df_stats['missing_pct'].mean():.2f}%")
    print(f"  Máximo: {df_stats['missing_pct'].max():.2f}%")
    print(f"  Datasets con valores faltantes: {(df_stats['missing_pct'] > 0).sum()}")
    
    if 'n_classes' in df_stats.columns:
        valid_classes = df_stats['n_classes'].dropna()
        if len(valid_classes) > 0:
            print(f"\nClases:")
            print(f"  Promedio: {valid_classes.mean():.1f}")
            print(f"  Rango: {valid_classes.min():.0f} - {valid_classes.max():.0f}")
    
    if 'has_benchmark' in df_stats.columns:
        print(f"\nBenchmarks:")
        print(f"  Datasets con benchmarks: {df_stats['has_benchmark'].sum()}")
    
    return df_stats


def main():
    """Función principal"""
    # Directorios - asegurar que son relativos al directorio raíz del proyecto
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "AEON" / "data"
    
    # Analizar datasets de clasificación
    classification_df = analyze_classification_datasets(
        data_dir,
        max_datasets=None,  # None = todos
        save_datasets=True
    )
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print(f"\nResultados guardados en: {data_dir}")
    print(f"Total de datasets analizados: {len(classification_df)}")
    print(f"Archivo JSON: {data_dir / 'classification_stats.json'}")


if __name__ == "__main__":
    main()


"""
Filtra datasets según criterios específicos
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict

from src.data_loader import TimeSeriesDataset

def filter_datasets(
    stats_json_path: Path,
    datasets_pkl_path: Path,
    output_json_path: Path,
    output_pkl_path: Path
):
    """
    Filtra datasets según criterios:
    - n_samples (train + test) <= 10000
    - n_dimensions (features) <= 15
    - length (steps) <= 1000
    - n_classes <= 10
    """
    # Cargar estadísticas
    print("Cargando estadísticas...")
    with open(stats_json_path, 'r') as f:
        all_stats = json.load(f)
    
    # Cargar datasets
    print("Cargando datasets...")
    with open(datasets_pkl_path, 'rb') as f:
        all_datasets = pickle.load(f)
    
    # Crear diccionario de datasets por nombre
    datasets_dict = {ds.name: ds for ds in all_datasets}
    
    # Filtrar datasets
    print("\nFiltrando datasets...")
    filtered_stats = []
    filtered_names = []
    
    for stats in all_stats:
        name = stats['name']
        
        # Obtener valores
        n_samples = stats.get('n_samples', 0)
        n_dimensions = stats.get('n_dimensions', 0)
        length = stats.get('length', 0)
        n_classes = stats.get('n_classes', None)
        
        # Aplicar filtros
        if (n_samples <= 10000 and 
            n_dimensions <= 15 and 
            length <= 1000 and 
            (n_classes is None or n_classes <= 10)):
            filtered_stats.append(stats)
            filtered_names.append(name)
    
    print(f"\nTotal de datasets: {len(all_stats)}")
    print(f"Datasets que cumplen criterios: {len(filtered_stats)}")
    
    # Guardar nombres en JSON
    print(f"\nGuardando nombres en {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(filtered_names, f, indent=2)
    
    # Calcular estadísticas de los filtrados
    if filtered_stats:
        df = pd.DataFrame(filtered_stats)
        
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS DE DATASETS FILTRADOS")
        print("=" * 80)
        
        print(f"\nTotal de datasets filtrados: {len(filtered_stats)}")
        
        print(f"\nNúmero de muestras (train + test):")
        print(f"  Min: {df['n_samples'].min()}")
        print(f"  Max: {df['n_samples'].max()}")
        print(f"  Mean: {df['n_samples'].mean():.1f}")
        print(f"  Median: {df['n_samples'].median():.1f}")
        
        print(f"\nNúmero de dimensiones (features):")
        print(f"  Min: {df['n_dimensions'].min()}")
        print(f"  Max: {df['n_dimensions'].max()}")
        print(f"  Mean: {df['n_dimensions'].mean():.1f}")
        print(f"  Median: {df['n_dimensions'].median():.1f}")
        
        print(f"\nLongitud temporal (steps):")
        print(f"  Min: {df['length'].min()}")
        print(f"  Max: {df['length'].max()}")
        print(f"  Mean: {df['length'].mean():.1f}")
        print(f"  Median: {df['length'].median():.1f}")
        
        valid_classes = df['n_classes'].dropna()
        if len(valid_classes) > 0:
            print(f"\nNúmero de clases:")
            print(f"  Min: {valid_classes.min()}")
            print(f"  Max: {valid_classes.max()}")
            print(f"  Mean: {valid_classes.mean():.1f}")
            print(f"  Median: {valid_classes.median():.1f}")
        
        print(f"\nDistribución completa:")
        print(df[['n_samples', 'n_dimensions', 'length', 'n_classes']].describe())
    
    # Filtrar datasets del pkl
    print(f"\nGuardando datasets filtrados en {output_pkl_path}...")
    filtered_datasets = [datasets_dict[name] for name in filtered_names if name in datasets_dict]
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(filtered_datasets, f)
    
    print(f"Guardados {len(filtered_datasets)} datasets filtrados")
    
    return filtered_names, filtered_stats


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    data_dir = base_dir / "AEON" / "data"
    
    stats_json_path = data_dir / "classification_stats.json"
    datasets_pkl_path = data_dir / "classification_datasets.pkl"
    output_json_path = data_dir / "filtered_dataset_names.json"
    output_pkl_path = data_dir / "classification_datasets.pkl"  # Sobrescribir el original
    
    filter_datasets(
        stats_json_path,
        datasets_pkl_path,
        output_json_path,
        output_pkl_path
    )




"""
Statistical analysis for time series datasets
"""
import sys
from pathlib import Path

# Agregar el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesStatistics:
    """Calcula estadísticas para datasets de series temporales"""
    
    @staticmethod
    def compute_dataset_stats_simple(dataset) -> Dict:
        """
        Calcula estadísticas básicas y ligeras para un dataset
        Extrae información directamente de los datos, no de metadata
        
        Args:
            dataset: TimeSeriesDataset object
            
        Returns:
            Dict con estadísticas básicas
        """
        # Usar train si está disponible, sino usar X combinado
        if dataset.X_train is not None:
            X = dataset.X_train  # Usar train para estadísticas
            n_train = dataset.n_train
            n_test = dataset.n_test
            n_total = n_train + n_test
        else:
            X = dataset.X
            n_train = None
            n_test = None
            n_total = dataset.n
        
        # Formato de aeon: (n, channels, length)
        # n = número de muestras
        # channels = número de dimensiones/variables/canales
        # length = longitud temporal (timesteps)
        if X is not None and X.size > 0:
            n_samples_in_X, channels, length = X.shape
        else:
            channels, length = 0, 0
            n_samples_in_X = 0
        
        # Detectar valores faltantes/padding
        # Los valores faltantes pueden ser NaN o '?' (que aeon puede convertir a NaN o mantener como string)
        # Para series de variable length en formato ARFF, están paddeadas con '?' que aeon convierte a NaN
        # Weka ARFF format: todas las series de longitud desigual están padded con missing values ('?')
        # aeon ts format: permite series de longitud desigual sin padding
        
        # Detectar valores faltantes: NaN o valores especiales
        # aeon generalmente convierte '?' a NaN, pero verificamos ambos casos
        missing_mask = np.isnan(X)
        
        # Si hay strings, buscar '?' u otros valores especiales
        if X.dtype == object or (hasattr(X.dtype, 'kind') and X.dtype.kind == 'U'):
            try:
                # Convertir a string y buscar '?'
                X_str = X.astype(str)
                missing_mask = missing_mask | (X_str == '?') | (X_str == 'nan') | (X_str == '') | (X_str == 'None')
            except:
                pass
        
        missing_pct = float(missing_mask.sum() / (n_samples_in_X * channels * length) * 100) if (n_samples_in_X * channels * length) > 0 else 0.0
        
        # Calcular longitudes reales de cada serie (excluyendo valores faltantes/padding)
        # Formato aeon: (n, channels, length)
        # Para cada muestra, contar cuántos timesteps son válidos (no NaN, no '?')
        real_lengths = []
        for i in range(n_samples_in_X):
            # Para cada canal, contar timesteps válidos
            # Generalmente todos los canales tienen la misma longitud real
            valid_timesteps = []
            for ch in range(channels):
                valid = ~missing_mask[i, ch, :]  # Formato (n, channels, length)
                valid_timesteps.append(valid.sum())
            # La longitud real es el máximo entre canales (si hay diferencias, usar el máximo)
            real_length = max(valid_timesteps) if valid_timesteps else 0
            real_lengths.append(real_length)
        
        min_length = int(min(real_lengths)) if real_lengths and min(real_lengths) > 0 else length
        max_length = int(max(real_lengths)) if real_lengths and max(real_lengths) > 0 else length
        
        # Detectar variable length:
        # 1. Si min_length != max_length, tiene series de longitud variable
        # 2. Si metadata dice equallength=False, es variable length
        # 3. Si hay muchos valores faltantes al final de las series, probablemente es padding de variable length
        has_variable_length = False
        
        if dataset.metadata and 'equallength' in dataset.metadata:
            # Si metadata dice equallength=False, es variable length
            has_variable_length = not dataset.metadata['equallength']
        elif min_length != max_length:
            # Si las longitudes reales difieren, es variable length
            has_variable_length = True
        elif missing_pct > 0:
            # Si hay valores faltantes, verificar si están al final (padding) o distribuidos (valores faltantes reales)
            # Para variable length, el padding suele estar al final de las series
            # Formato aeon: (n, channels, length)
            # Contar cuántas series tienen NaN al final (último 20% de timesteps)
            samples_with_end_padding = 0
            for i in range(n):
                for ch in range(channels):
                    series = missing_mask[i, ch, :]  # Formato (n, channels, length)
                    if series.sum() > 0:
                        # Verificar si los NaN están principalmente al final
                        last_portion = int(length * 0.2)  # Último 20%
                        if last_portion > 0:
                            end_missing = series[-last_portion:].sum()
                            total_missing = series.sum()
                            # Si más del 50% de los NaN están al final, probablemente es padding
                            if total_missing > 0 and (end_missing / total_missing) > 0.5:
                                samples_with_end_padding += 1
                                break
            
            # Si más del 30% de las series tienen padding al final, probablemente es variable length
            if n_samples_in_X > 0 and (samples_with_end_padding / n_samples_in_X) > 0.3:
                has_variable_length = True
        
        stats_dict = {
            'name': dataset.name,
            'shape': f"({n_total}, {channels}, {length})",  # (samples, channels, length) - formato aeon
            'n_samples': n_total,
            'length': length,  # Longitud temporal máxima (puede estar padded para variable length)
            'min_length': min_length,  # Longitud mínima real (sin padding)
            'max_length': max_length,  # Longitud máxima real (sin padding)
            'n_dimensions': channels,  # Número de dimensiones/variables/canales
            'missing_pct': missing_pct,
            'has_variable_length': has_variable_length,
        }
        
        # Contar muestras con valores faltantes (puede ser padding o valores faltantes reales)
        # Formato (n, channels, length)
        nan_per_sample = missing_mask.sum(axis=(1, 2))
        samples_with_missing = (nan_per_sample > 0).sum()
        stats_dict['samples_with_missing'] = int(samples_with_missing)
        
        # Estadísticas básicas por dimensión (mean, std, min, max)
        # Formato aeon: (n, channels, length)
        dimension_stats = []
        for ch in range(channels):
            ch_data = X[:, ch, :]  # Canal ch: (n, length)
            ch_missing = missing_mask[:, ch, :]
            valid_data = ch_data[~ch_missing]
            
            if len(valid_data) > 0:
                # Convertir a float si es necesario
                try:
                    valid_data_float = valid_data.astype(float)
                    dimension_stats.append({
                        'dimension': int(ch),
                        'mean': float(np.mean(valid_data_float)),
                        'std': float(np.std(valid_data_float)),
                        'min': float(np.min(valid_data_float)),
                        'max': float(np.max(valid_data_float)),
                    })
                except (ValueError, TypeError):
                    # Si no se puede convertir a float, usar valores como están
                    dimension_stats.append({
                        'dimension': int(ch),
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                    })
            else:
                dimension_stats.append({
                    'dimension': int(ch),
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                })
        stats_dict['dimension_stats'] = dimension_stats
        
        # Info de targets y clases (usar train+test si están separados)
        if dataset.X_train is not None:
            # Combinar y_train y y_test para calcular clases
            if dataset.y_train is not None and dataset.y_test is not None:
                y = np.concatenate([dataset.y_train, dataset.y_test])
            elif dataset.y_train is not None:
                y = dataset.y_train
            elif dataset.y_test is not None:
                y = dataset.y_test
            else:
                y = None
        else:
            y = dataset.y
        
        if y is not None:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            
            stats_dict['n_classes'] = int(n_classes)
            
            # Calcular balance de clases
            if len(y) > 0:
                unique, counts = np.unique(y, return_counts=True)
                class_balance = float(np.min(counts) / np.max(counts)) if len(counts) > 0 else 0.0
                stats_dict['class_balance'] = class_balance
        else:
            stats_dict['n_classes'] = None
        
        # Train/Test split - usar directamente del dataset
        if dataset.X_train is not None:
            stats_dict['train_size'] = int(n_train)
            stats_dict['test_size'] = int(n_test) if n_test is not None else None
        elif dataset.metadata:
            train_size = dataset.metadata.get('train_size')
            test_size = dataset.metadata.get('test_size')
            stats_dict['train_size'] = int(train_size) if train_size is not None else None
            stats_dict['test_size'] = int(test_size) if test_size is not None else None
        else:
            stats_dict['train_size'] = None
            stats_dict['test_size'] = None
        
        return stats_dict
    
    @staticmethod
    def compute_dataset_stats(dataset, sample_size: int = 10) -> Dict:
        """
        Calcula estadísticas para un dataset completo
        
        Args:
            dataset: TimeSeriesDataset object
            sample_size: Número de muestras para análisis de estacionariedad
        """
        X = dataset.X
        n, s, m = X.shape
        
        stats_dict = {
            'name': dataset.name,
            'n_samples': n,
            'n_timesteps': s,
            'n_channels': m,
            'total_values': n * s * m,
            'missing_values': int(np.isnan(X).sum()),
            'missing_pct': float(np.isnan(X).sum() / (n * s * m) * 100),
            'dtype': str(X.dtype),
        }
        
        # Estadísticas por canal
        channel_stats = []
        for ch in range(m):
            channel_data = X[:, :, ch]
            valid_data = channel_data[~np.isnan(channel_data)]
            
            if len(valid_data) > 0:
                channel_stats.append({
                    'channel': int(ch),
                    'mean': float(np.nanmean(channel_data)),
                    'std': float(np.nanstd(channel_data)),
                    'min': float(np.nanmin(channel_data)),
                    'max': float(np.nanmax(channel_data)),
                    'median': float(np.nanmedian(channel_data)),
                    'q25': float(np.nanpercentile(channel_data, 25)),
                    'q75': float(np.nanpercentile(channel_data, 75)),
                    'skewness': float(stats.skew(valid_data)),
                    'kurtosis': float(stats.kurtosis(valid_data)),
                    'missing_pct': float(np.isnan(channel_data).sum() / (n * s) * 100),
                })
            else:
                channel_stats.append({
                    'channel': int(ch),
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'missing_pct': 100.0,
                })
        
        stats_dict['channel_stats'] = channel_stats
        
        # Estadísticas temporales
        # Verificar si hay series de diferentes longitudes
        if X.ndim == 3:
            # Para datasets con series de igual longitud
            temporal_stats = {
                'mean_length': int(s),
                'min_length': int(s),
                'max_length': int(s),
                'length_variance': 0.0,
                'equal_length': True,
            }
        else:
            temporal_stats = {
                'mean_length': float(np.mean([len(series) for series in X])),
                'min_length': int(np.min([len(series) for series in X])),
                'max_length': int(np.max([len(series) for series in X])),
                'length_variance': float(np.var([len(series) for series in X])),
                'equal_length': False,
            }
        
        stats_dict['temporal_stats'] = temporal_stats
        
        # Estadísticas de estacionariedad (muestra representativa)
        stationarity_stats = []
        sample_indices = np.linspace(0, n-1, min(sample_size, n), dtype=int)
        
        for idx in sample_indices:
            for ch in range(min(3, m)):  # Limitar a 3 canales para no sobrecargar
                series = X[idx, :, ch]
                valid_series = series[~np.isnan(series)]
                
                if len(valid_series) > 10:  # Mínimo para test de estacionariedad
                    try:
                        adf_result = adfuller(valid_series, autolag='AIC')
                        stationarity_stats.append({
                            'sample': int(idx),
                            'channel': int(ch),
                            'adf_statistic': float(adf_result[0]),
                            'adf_pvalue': float(adf_result[1]),
                            'is_stationary': adf_result[1] < 0.05,
                            'n_observations': len(valid_series),
                        })
                    except Exception as e:
                        pass
        
        stats_dict['stationarity'] = stationarity_stats
        stats_dict['n_stationarity_tests'] = len(stationarity_stats)
        
        # Correlación entre canales (promedio temporal)
        if m > 1:
            try:
                mean_series = np.nanmean(X, axis=1)  # (n, m)
                # Remover filas con NaN
                valid_rows = ~np.isnan(mean_series).any(axis=1)
                if valid_rows.sum() > 1:
                    channel_corr = np.corrcoef(mean_series[valid_rows].T)
                    stats_dict['channel_correlation'] = channel_corr.tolist()
                    stats_dict['channel_correlation_mean'] = float(np.nanmean(channel_corr[np.triu_indices(m, k=1)]))
                else:
                    stats_dict['channel_correlation'] = None
                    stats_dict['channel_correlation_mean'] = np.nan
            except:
                stats_dict['channel_correlation'] = None
                stats_dict['channel_correlation_mean'] = np.nan
        else:
            stats_dict['channel_correlation'] = None
            stats_dict['channel_correlation_mean'] = np.nan
        
        # Estadísticas de targets si existen
        if dataset.y is not None:
            y = dataset.y
            target_stats = {
                'has_targets': True,
                'target_dtype': str(y.dtype),
            }
            
            # Verificar si es numérico o string
            is_numeric = np.issubdtype(y.dtype, np.number)
            is_integer = np.issubdtype(y.dtype, np.integer) if is_numeric else False
            
            # Si es string o object, tratar como clasificación
            if not is_numeric or is_integer:
                # Clasificación (puede ser integer o string)
                try:
                    unique, counts = np.unique(y, return_counts=True)
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = int(len(unique))
                    # Convertir a dict, manejando strings e integers
                    class_dist = {}
                    for k, v in zip(unique, counts):
                        if isinstance(k, (str, np.str_)):
                            class_dist[str(k)] = int(v)
                        else:
                            class_dist[int(k)] = int(v)
                    target_stats['class_distribution'] = class_dist
                    target_stats['class_balance'] = float(np.min(counts) / np.max(counts)) if len(counts) > 0 else 0.0
                except Exception as e:
                    # Si falla, solo guardar info básica
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = len(np.unique(y)) if hasattr(y, '__len__') else None
            else:
                # Regresión (numérico pero no integer)
                try:
                    y_numeric = y.astype(float)
                    target_stats['target_type'] = 'regression'
                    target_stats['mean'] = float(np.nanmean(y_numeric))
                    target_stats['std'] = float(np.nanstd(y_numeric))
                    target_stats['min'] = float(np.nanmin(y_numeric))
                    target_stats['max'] = float(np.nanmax(y_numeric))
                except Exception as e:
                    # Si falla la conversión, tratar como clasificación
                    unique, counts = np.unique(y, return_counts=True)
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = int(len(unique))
            
            stats_dict['target_stats'] = target_stats
        else:
            stats_dict['target_stats'] = {'has_targets': False}
        
        return stats_dict
    
    @staticmethod
    def compute_global_stats(datasets: List) -> pd.DataFrame:
        """
        Agrega estadísticas de múltiples datasets
        
        Args:
            datasets: Lista de TimeSeriesDataset objects
            
        Returns:
            DataFrame con estadísticas agregadas
        """
        all_stats = []
        
        for dataset in datasets:
            stats_dict = TimeSeriesStatistics.compute_dataset_stats(dataset)
            
            # Extraer estadísticas de canales
            channel_means = [ch['mean'] for ch in stats_dict['channel_stats'] if not np.isnan(ch['mean'])]
            channel_stds = [ch['std'] for ch in stats_dict['channel_stats'] if not np.isnan(ch['std'])]
            
            row = {
                'name': stats_dict['name'],
                'n_samples': stats_dict['n_samples'],
                'n_timesteps': stats_dict['n_timesteps'],
                'n_channels': stats_dict['n_channels'],
                'missing_pct': stats_dict['missing_pct'],
                'mean_value': np.mean(channel_means) if channel_means else np.nan,
                'std_value': np.mean(channel_stds) if channel_stds else np.nan,
                'channel_corr_mean': stats_dict.get('channel_correlation_mean', np.nan),
                'n_stationarity_tests': stats_dict.get('n_stationarity_tests', 0),
            }
            
            # Agregar info de targets
            if stats_dict['target_stats']['has_targets']:
                row['has_targets'] = True
                row['target_type'] = stats_dict['target_stats'].get('target_type', 'unknown')
                if row['target_type'] == 'classification':
                    row['n_classes'] = stats_dict['target_stats'].get('n_classes', np.nan)
                else:
                    row['n_classes'] = np.nan
            else:
                row['has_targets'] = False
                row['target_type'] = None
                row['n_classes'] = np.nan
            
            all_stats.append(row)
        
        return pd.DataFrame(all_stats)


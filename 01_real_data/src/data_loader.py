"""
Data loader for time series datasets
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset:
    """Wrapper para datasets de series temporales con train y test separados"""
    
    def __init__(self, name: str, 
                 X_train: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
        Args:
            name: Nombre del dataset
            X_train: Array de train de forma (n_train, s, m)
            y_train: Targets de train de forma (n_train,)
            X_test: Array de test de forma (n_test, s, m)
            y_test: Targets de test de forma (n_test,)
            X: Array combinado (si no se proporciona train/test por separado)
            y: Targets combinados (si no se proporciona train/test por separado)
            metadata: Diccionario con metadatos adicionales
        """
        self.name = name
        self.metadata = metadata or {}
        
        # Si se proporciona train/test por separado, usarlos
        if X_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            # Asegurar que son 3D
            if X_train.ndim == 2:
                self.X_train = X_train[:, :, np.newaxis]
            if X_test is not None and X_test.ndim == 2:
                self.X_test = X_test[:, :, np.newaxis]
            
            self.n_train, self.s_train, self.m_train = self.X_train.shape
            if self.X_test is not None:
                self.n_test, self.s_test, self.m_test = self.X_test.shape
            else:
                self.n_test, self.s_test, self.m_test = 0, 0, 0
            
            # Para compatibilidad, X e y apuntan a train+test combinados
            if X_test is not None:
                self.X = np.concatenate([self.X_train, self.X_test], axis=0)
                self.y = np.concatenate([self.y_train, self.y_test], axis=0) if (y_train is not None and y_test is not None) else None
            else:
                self.X = self.X_train
                self.y = self.y_train
                
            self.n, self.s, self.m = self.X.shape
            
        else:
            # Modo legacy: solo X e y combinados
            self.X = X
            self.y = y
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            
            if X is not None:
                if X.ndim == 2:
                    self.X = X[:, :, np.newaxis]
                self.n, self.s, self.m = self.X.shape
            else:
                self.n, self.s, self.m = 0, 0, 0
        
    def __repr__(self):
        if self.X_train is not None:
            return f"TimeSeriesDataset(name='{self.name}', train={self.X_train.shape}, test={self.X_test.shape if self.X_test is not None else None})"
        else:
            return f"TimeSeriesDataset(name='{self.name}', shape={self.X.shape if self.X is not None else None}, has_targets={self.y is not None})"
    
    def get_info(self) -> Dict:
        """Retorna información básica del dataset"""
        info = {
            'name': self.name,
            'has_train_test_split': self.X_train is not None,
        }
        
        if self.X_train is not None:
            info['n_train'] = self.n_train
            info['n_test'] = self.n_test
            info['n_timesteps'] = self.s_train
            info['n_channels'] = self.m_train
            info['has_targets'] = self.y_train is not None
            
            if self.y_train is not None:
                if np.issubdtype(self.y_train.dtype, np.integer):
                    info['target_type'] = 'classification'
                    info['n_classes'] = len(np.unique(np.concatenate([self.y_train, self.y_test]) if self.y_test is not None else self.y_train))
                else:
                    info['target_type'] = 'regression'
        else:
            info['n_samples'] = self.n
            info['n_timesteps'] = self.s
            info['n_channels'] = self.m
            info['has_targets'] = self.y is not None
            
            if self.y is not None:
                if np.issubdtype(self.y.dtype, np.integer):
                    info['target_type'] = 'classification'
                    info['n_classes'] = len(np.unique(self.y))
                else:
                    info['target_type'] = 'regression'
        
        return info


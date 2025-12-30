"""
Load and analyze classification datasets from UCR/UEA archive via aeon
"""
import sys
from pathlib import Path

# Agregar el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Optional
import warnings
from tqdm import tqdm
import pickle

try:
    from aeon.datasets import load_classification
    # Intentar diferentes formas de obtener la lista de datasets
    try:
        from aeon.datasets import get_dataset_names
    except ImportError:
        try:
            from aeon.datasets import list_available_datasets
            get_dataset_names = list_available_datasets
        except ImportError:
            # Si no existe, usaremos una lista conocida
            get_dataset_names = None
except ImportError:
    print("Warning: aeon not installed. Install with: pip install aeon")
    load_classification = None
    get_dataset_names = None

from src.data_loader import TimeSeriesDataset

warnings.filterwarnings('ignore')


# Lista completa de datasets UCR/UEA de clasificación
# Fuente: https://www.timeseriesclassification.com/dataset.php
UCR_CLASSIFICATION_DATASETS = [
    "AbnormalHeartbeat", "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "ArrowHead", "ArticularyWordRecognition", "AsphaltObstacles",
    "AsphaltObstaclesCoordinates", "AsphaltPavementType", "AsphaltPavementTypeCoordinates",
    "AsphaltRegularity", "AsphaltRegularityCoordinates", "AtrialFibrillation", "BasicMotions",
    "Beef", "BeetleFly", "BinaryHeartbeat", "BirdChicken", "Blink", "BME", "Car",
    "CardiacArrhythmia", "CatsDogs", "CBF", "CharacterTrajectories", "Chinatown",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Colposcopy", "Computers",
    "CounterMovementJump", "Cricket", "CricketX", "CricketY", "CricketZ", "Crop",
    "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "DuckDuckGeese", "DucksAndGeese", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
    "EigenWorms", "ElectricDeviceDetection", "ElectricDevices", "EMOPain",
    "EOGHorizontalSignal", "EOGVerticalSignal", "Epilepsy", "Epilepsy2", "ERing",
    "EthanolConcentration", "EthanolLevel", "EyesOpenShut", "FaceAll", "FaceDetection",
    "FaceFour", "FacesUCR", "FaultDetectionA", "FaultDetectionB", "FiftyWords",
    "FingerMovements", "Fish", "FordA", "FordB", "FreezerRegularTrain", "FreezerSmallTrain",
    "FruitFlies", "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham", "HandMovementDirection",
    "HandOutlines", "Handwriting", "Haptics", "Heartbeat", "Herring", "HouseTwenty",
    "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectSound",
    "InsectWingbeat", "ItalyPowerDemand", "JapaneseVowels", "KeplerLightCurves",
    "LargeKitchenAppliances", "Libras", "Lightning2", "Lightning7", "LSST", "Mallat",
    "Meat", "MedicalImages", "MelbournePedestrian", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MindReading",
    "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MosquitoSound", "MoteStrain",
    "MotionSenseHAR", "MotorImagery", "NATOPS", "NerveDamage", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PEMS-SF", "PenDigits",
    "PhalangesOutlinesCorrect", "Phoneme", "PhonemeSpectra", "PickupGestureWiimoteZ",
    "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "Plane", "PowerCons",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
    "RacketSports", "RefrigerationDevices", "RightWhaleCalls", "Rock", "ScreenType",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll",
    "SharePriceIncrease", "Sleep", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "SpokenArabicDigits",
    "StandWalkJump", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
    "SyntheticControl", "Tiselac", "ToeSegmentation1", "ToeSegmentation2", "Trace",
    "TwoLeadECG", "TwoPatterns", "UMD", "UrbanSound", "UWaveGestureLibrary",
    "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "Wafer", "WalkingSittingStanding", "Wine", "WordSynonyms",
    "Worms", "WormsTwoClass", "Yoga",
]


def get_all_classification_datasets() -> List[str]:
    """
    Obtiene lista de todos los datasets de clasificación disponibles.
    Usa la lista completa proporcionada de https://www.timeseriesclassification.com/dataset.php
    """
    # Usar directamente la lista completa que proporcionaste
    return sorted(UCR_CLASSIFICATION_DATASETS)


def load_single_classification_dataset(name: str, verbose: bool = False) -> Optional[TimeSeriesDataset]:
    """
    Carga un dataset de clasificación
    
    Args:
        name: Nombre del dataset
        verbose: Si True, imprime información
        
    Returns:
        TimeSeriesDataset o None si hay error
    """
    if load_classification is None:
        print("aeon not installed")
        return None
    
    try:
        # Cargar train y test por separado
        X_train, y_train, metadata_train = load_classification(name, split="TRAIN", return_metadata=True)
        X_test, y_test, metadata_test = load_classification(name, split="test", return_metadata=True)
        
        # Guardar datos tal como vienen de aeon, sin transponer
        # aeon devuelve: (n, channels, length) para multivariado o (n, length) para univariado
        if X_train.ndim == 2:
            # Univariado: (n, length) -> mantener como está o agregar dimensión si es necesario
            # Para compatibilidad con TimeSeriesDataset que espera 3D, agregamos dimensión
            X_train = X_train[:, :, np.newaxis]
        
        if X_test.ndim == 2:
            # Univariado: (n, length) -> mantener como está o agregar dimensión si es necesario
            X_test = X_test[:, :, np.newaxis]
        
        # Combinar metadata (preferir train)
        metadata = metadata_train.copy() if metadata_train else {}
        if metadata_test:
            metadata.update(metadata_test)
        
        # Agregar información de train/test split
        metadata['train_size'] = len(X_train)
        metadata['test_size'] = len(X_test)
        
        if verbose:
            print(f"\nDataset: {name}")
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Crear dataset con train y test separados (NO combinados)
        dataset = TimeSeriesDataset(
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata
        )
        return dataset
        
    except Exception as e:
        if verbose:
            print(f"  Error loading {name}: {e}")
        return None


def load_all_classification_datasets(max_datasets: Optional[int] = None, 
                                     save_path: Optional[Path] = None,
                                     verbose: bool = True) -> List[TimeSeriesDataset]:
    """
    Carga todos los datasets de clasificación disponibles
    
    Args:
        max_datasets: Número máximo de datasets a cargar (None = todos)
        save_path: Path para guardar los datasets cargados
        verbose: Si True, muestra progreso
        
    Returns:
        Lista de TimeSeriesDataset objects
    """
    dataset_names = get_all_classification_datasets()
    
    if max_datasets:
        dataset_names = dataset_names[:max_datasets]
    
    if verbose:
        print(f"Loading {len(dataset_names)} classification datasets...")
    
    datasets = []
    failed = []
    
    for name in tqdm(dataset_names, desc="Loading datasets"):
        dataset = load_single_classification_dataset(name, verbose=False)
        if dataset is not None:
            datasets.append(dataset)
        else:
            failed.append(name)
    
    if verbose:
        print(f"\nSuccessfully loaded: {len(datasets)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print(f"Failed datasets: {failed[:10]}...")  # Mostrar primeros 10
    
    # Guardar si se especifica path
    if save_path and datasets:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(datasets, f)
        if verbose:
            print(f"\nSaved datasets to {save_path}")
    
    return datasets


if __name__ == "__main__":
    # Test loading a few datasets
    print("Testing classification dataset loading...")
    
    # Test single dataset
    test_dataset = load_single_classification_dataset("GunPoint", verbose=True)
    if test_dataset:
        print(f"\nTest dataset info: {test_dataset.get_info()}")
    
    # Load all (or a subset for testing)
    # datasets = load_all_classification_datasets(max_datasets=5, verbose=True)


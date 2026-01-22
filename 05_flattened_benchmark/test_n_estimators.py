"""
Quick test to compare TabPFN performance with different n_estimators.
Tests on a few synthetic datasets to see if n_estimators=1 explains the accuracy gap.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Paths
SYNTHETIC_DIR = Path(__file__).parent / "results" / "synthetic_datasets"

def test_n_estimators():
    from tabpfn import TabPFNClassifier
    
    # Load a few synthetic datasets
    npz_files = sorted(SYNTHETIC_DIR.glob("synthetic_*.npz"))[:5]
    
    print("=" * 60)
    print("COMPARING n_estimators EFFECT ON ACCURACY")
    print("=" * 60)
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        X = data['X']  # (n_samples, n_features, length)
        y = data['y']
        n_classes = int(data['n_classes'])
        
        # Flatten
        X_flat = X.reshape(X.shape[0], -1)
        
        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=0.3, stratify=y, random_state=42
            )
        except ValueError:
            print(f"{npz_file.stem}: Cannot stratify (skipping)")
            continue
        
        # Encode labels
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        
        print(f"\n{npz_file.stem}: {X_flat.shape[0]} samples, {X_flat.shape[1]} features, {n_classes} classes")
        
        # Test with n_estimators=1
        clf_1 = TabPFNClassifier(device='cpu', n_estimators=1, ignore_pretraining_limits=True)
        clf_1.fit(X_train.astype(np.float32), y_train_enc)
        y_pred_1 = clf_1.predict(X_test.astype(np.float32))
        acc_1 = accuracy_score(y_test_enc, y_pred_1)
        
        # Test with n_estimators=4
        clf_4 = TabPFNClassifier(device='cpu', n_estimators=4, ignore_pretraining_limits=True)
        clf_4.fit(X_train.astype(np.float32), y_train_enc)
        y_pred_4 = clf_4.predict(X_test.astype(np.float32))
        acc_4 = accuracy_score(y_test_enc, y_pred_4)
        
        # Test with n_estimators=16 (default-ish)
        clf_16 = TabPFNClassifier(device='cpu', n_estimators=16, ignore_pretraining_limits=True)
        clf_16.fit(X_train.astype(np.float32), y_train_enc)
        y_pred_16 = clf_16.predict(X_test.astype(np.float32))
        acc_16 = accuracy_score(y_test_enc, y_pred_16)
        
        print(f"  n_estimators=1:  Acc = {acc_1:.4f}")
        print(f"  n_estimators=4:  Acc = {acc_4:.4f}")
        print(f"  n_estimators=16: Acc = {acc_16:.4f}")
        print(f"  Difference (16 vs 1): {acc_16 - acc_1:+.4f}")


if __name__ == "__main__":
    test_n_estimators()

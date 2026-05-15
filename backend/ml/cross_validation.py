"""
cross_validation.py
===================
Time Series Cross Validation for more robust performance metrics.
"""
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from config.logging_config import get_logger
import numpy as np
import pandas as pd

logger = get_logger(__name__)

def time_series_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Evaluates a model using TimeSeriesSplit.
    Data should be ordered chronologically *before* passing to this function.
    
    Args:
        model: An untaught classifier (scikit-learn API)
        X: Feature matrix
        y: Target array
        n_splits: Number of folds
        
    Returns:
        Dictionary with mean and std of accuracy and AUC
    """
    logger.info(f"Iniciando TimeSeriesSplit Cross Validation ({n_splits} folds)...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    accuracies = []
    aucs = []
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    from sklearn.base import clone
    
    fold = 1
    for train_index, test_index in tscv.split(X_arr):
        logger.debug(f"Fold {fold}/{n_splits} - Train: {len(train_index)}, Test: {len(test_index)}")
        
        X_tr, X_te = X_arr[train_index], X_arr[test_index]
        y_tr, y_te = y_arr[train_index], y_arr[test_index]
        
        # Clone the model to ensure fresh start each fold
        fold_model = clone(model)
        
        # Fit
        if hasattr(fold_model, 'fit') and 'verbose' in fold_model.get_params():
            fold_model.fit(X_tr, y_tr, verbose=False)
        else:
            fold_model.fit(X_tr, y_tr)
            
        # Predict
        preds = fold_model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        accuracies.append(acc)
        
        # Probas for AUC (if available)
        if hasattr(fold_model, 'predict_proba'):
            probas = fold_model.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, probas)
                aucs.append(auc)
            except Exception:
                pass
                
        logger.debug(f"  → Accuracy: {acc:.4f}")
        fold += 1
        
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    results = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }
    
    if aucs:
        results["mean_auc"] = np.mean(aucs)
        results["std_auc"] = np.std(aucs)
    
    logger.info(f"CV Completado. Mean Acc: {mean_acc:.4f} (±{std_acc:.4f})")
    
    return results

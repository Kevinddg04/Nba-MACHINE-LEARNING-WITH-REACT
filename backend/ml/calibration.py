"""
calibration.py
==============
Probability calibration (Platt scaling) for the classifier.
"""
from sklearn.calibration import CalibratedClassifierCV
from config.logging_config import get_logger

logger = get_logger(__name__)

def calibrate_classifier(model, X_train, y_train, cv='prefit'):
    """
    Applies Platt scaling (sigmoid calibration) to improve the 
    reliability of predicted probabilities.
    
    Args:
        model: Trained classifier (CatBoost)
        X_train: Training data or Validation data
        y_train: Labels
        cv: Cross validation strategy. Use 'prefit' if passing validation data,
            or an integer (e.g. 5) if passing training data to retrain folds.
            
    Returns:
        calibrated_model
    """
    logger.info("Aplicando Calibración de Probabilidades (Platt Scaling)...")
    
    calibrated_model = CalibratedClassifierCV(
        estimator=model, 
        method='sigmoid', 
        cv=cv
    )
    
    calibrated_model.fit(X_train, y_train)
    logger.info("Calibración completada.")
    
    return calibrated_model

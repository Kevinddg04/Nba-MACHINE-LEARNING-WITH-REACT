"""
ensemble.py
===========
Construye el ensamble de clasificadores: CatBoost, LightGBM y Regresion Logistica.
Usa VotingClassifier con votacion suave (promedio de probabilidades).
"""
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from config.logging_config import get_logger

logger = get_logger(__name__)

def build_ensemble(catboost_params, lgbm_params):
    """
    Construye un VotingClassifier con 3 estimadores base.
    Los imports pesados (CatBoost, LightGBM) son lazy para no crashear en CI.
    """
    logger.info("Construyendo Ensemble (CatBoost + LightGBM + LogisticRegression)...")
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("CatBoost no está instalado. Ejecuta: pip install catboost")
        
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM no está instalado. Ejecuta: pip install lightgbm")
        
    # 1. CatBoost (Peso: 50%)
    cb_model = CatBoostClassifier(**catboost_params)
    
    # 2. LightGBM (Peso: 30%) 
    lgbm_model = LGBMClassifier(**lgbm_params)
    
    # 3. Regresión Logística (Peso: 20%) — Captura patrones lineales con probabilidades nativas
    from sklearn.linear_model import LogisticRegression
    ridge_proxy = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)
    
    # Ensamble por Votación Suave
    ensemble = VotingClassifier(
        estimators=[
            ('catboost', cb_model),
            ('lightgbm', lgbm_model),
            ('ridge', ridge_proxy)
        ],
        voting='soft',
        weights=[0.5, 0.3, 0.2]
    )
    
    return ensemble

def train_ensemble(X_train, y_train, catboost_params=None):
    """
    Trains the ensemble model. Note that early stopping is tricky with VotingClassifier,
    so we train with fixed iterations/trees.
    """
    if catboost_params is None:
        catboost_params = {
            'iterations': 1000, 
            'depth': 6, 
            'learning_rate': 0.03, 
            'l2_leaf_reg': 20.0, 
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.7,
            'random_seed': 42,
            'verbose': False
        }
        
    lgbm_params = {
        'n_estimators': 800,
        'learning_rate': 0.03,
        'max_depth': 6,
        'reg_lambda': 10.0, # L2 regularization equivalent
        'subsample': 0.7,
        'random_state': 42,
        'verbose': -1
    }
    
    ensemble = build_ensemble(catboost_params, lgbm_params)
    
    logger.info("Entrenando Ensemble (puede tardar un par de minutos)...")
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble entrenado exitosamente.")
    
    return ensemble

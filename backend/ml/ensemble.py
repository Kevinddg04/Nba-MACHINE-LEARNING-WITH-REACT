"""
ensemble.py
===========
Builds an ensemble classifier incorporating CatBoost, LightGBM, and Ridge.
VotingClassifier uses soft voting (probability averaging).
"""
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from config.logging_config import get_logger
from catboost import CatBoostClassifier

logger = get_logger(__name__)

def build_ensemble(catboost_params, lgbm_params):
    """
    Constructs a VotingClassifier with 3 base estimators.
    Requires lightgbm to be installed.
    """
    logger.info("Construyendo Ensemble (CatBoost + LightGBM + Ridge)...")
    
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.error("LightGBM no está instalado. Ejecuta: pip install lightgbm")
        raise
        
    # 1. CatBoost (Weight: 50%)
    cb_model = CatBoostClassifier(**catboost_params)
    
    # 2. LightGBM (Weight: 30%)
    lgbm_model = LGBMClassifier(**lgbm_params)
    
    # 3. Ridge (Weight: 20%) - Good for linear patterns and regularization
    # RidgeClassifier doesn't natively support predict_proba, but we can wrap it or
    # CalibratedClassifierCV does it automatically in soft voting sometimes.
    # Actually, RidgeClassifier doesn't output probas. Let's use LogisticRegression 
    # with L2 penalty instead since it gives probabilities naturally.
    from sklearn.linear_model import LogisticRegression
    ridge_proxy = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)
    
    # Voting Classifier
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

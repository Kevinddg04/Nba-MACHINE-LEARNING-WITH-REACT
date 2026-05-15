"""
feature_selection.py
====================
Provides tools for selecting the most informative features
and reducing multicollinearity in the ML pipeline.
"""
from sklearn.feature_selection import f_classif, SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config.logging_config import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)


def check_multicollinearity(X: pd.DataFrame, threshold: float = 10.0) -> list:
    """
    Calculates Variance Inflation Factor (VIF) to detect multicollinearity.
    Returns a list of columns to drop to keep VIF below threshold.
    """
    logger.info("Verificando multicolinealidad con VIF...")
    
    # Drop features containing 'label', 'teamId', 'gameId' etc. if they slipped in
    numeric_df = X.select_dtypes(include=[np.number])
    X_vif = numeric_df.copy()
    
    # We must drop target/label and identifiers
    cols_to_drop = []
    
    while True:
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        # Add a tiny constant to avoid division by zero
        vif_values = []
        for i in range(X_vif.shape[1]):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
            except Exception:
                vif = float('inf')
            vif_values.append(vif)
        
        vif_data["VIF"] = vif_values
        
        # Check max VIF
        max_vif_idx = vif_data["VIF"].idxmax()
        max_vif = vif_data.loc[max_vif_idx, "VIF"]
        
        if max_vif > threshold:
            feature_to_drop = vif_data.loc[max_vif_idx, "feature"]
            logger.debug(f"Removiendo {feature_to_drop} (VIF={max_vif:.2f})")
            X_vif = X_vif.drop(columns=[feature_to_drop])
            cols_to_drop.append(feature_to_drop)
        else:
            break
            
    logger.info(f"Multicolinealidad resuelta. Columnas removidas: {len(cols_to_drop)}")
    return cols_to_drop


def univariate_feature_selection(X: pd.DataFrame, y: pd.Series, k: int = 30) -> list:
    """
    Performs univariate feature selection (ANOVA F-value) to keep top K features.
    """
    logger.info(f"Seleccionando top {k} features con SelectKBest (ANOVA F-value)...")
    if len(X.columns) <= k:
        return list(X.columns)
        
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    selected_mask = selector.get_support()
    selected_cols = X.columns[selected_mask].tolist()
    
    # Identify dropped cols for logging
    dropped_cols = [c for c in X.columns if c not in selected_cols]
    logger.info(f"Features descartados ({len(dropped_cols)}): {dropped_cols}")
    
    return selected_cols


def select_features_final(X: pd.DataFrame, y: pd.Series, k: int = 35, vif_threshold: float = 10.0) -> list:
    """
    Full pipeline for feature selection:
    1. VIF reduction
    2. Univariate selection (SelectKBest)
    """
    logger.info(f"Iniciando selección de features... (Inicial: {X.shape[1]})")
    
    # 1. Univariate Selection to narrow down first (faster than VIF on many cols)
    top_cols = univariate_feature_selection(X, y, k=min(k + 10, X.shape[1]))
    X_reduced = X[top_cols]
    
    # 2. VIF for Multicollinearity reduction
    # Sometimes VIF is too aggressive on symmetric features like expectedTeamScore_A and B,
    # so we'll increase the threshold slightly and only run it if we still have many features
    cols_to_drop_vif = []
    if X_reduced.shape[1] > 10:
        cols_to_drop_vif = check_multicollinearity(X_reduced, threshold=vif_threshold)
    
    final_cols = [c for c in top_cols if c not in cols_to_drop_vif]
    
    # 3. Final Univariate trim if still too many
    if len(final_cols) > k:
        final_cols = univariate_feature_selection(X[final_cols], y, k=k)
        
    logger.info(f"Selección final: {len(final_cols)} features mantenidos.")
    return final_cols

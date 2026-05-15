# Arquitectura del Sistema: NBA ML Predictor v2.0

## Diagrama Funcional
```
[Frontend - React/TypeScript]
       |
     (REST/JSON via Axios)
       |
[Backend - FastAPI (Python 3.11)] ---> [Caché: Redis (Alpine)]
       |
       +--> [Auditoría: SQLite/SQLAlchemy]
       |
[ML CorePipeline (pandas, scikit-learn)]
       |
       +--> [Ensemble: CatBoost + LightGBM + Ridge]
```

## Resumen del Flujo ML
1. **Adquisición**: `kaggle_fetcher.py` extrae un CSV actualizado de los box scores NBA usando la API de Kaggle.
2. **Ingeniería**: `ml_pipeline.py` procesa los box scores calculando ventanas móviles (3, 10 y 20 juegos) y deduciendo un `RealHandicap`.
3. **Reducción**: `ml/feature_selection.py` destruye dependencias lineales con Factor de Inflación de Varianza (VIF) y elige las mejores ~35 columnas empleando SelectKBest (ANOVA).
4. **Ensamble**: Se construye un `VotingClassifier` ("soft") compuesto de CatBoost (50%), LightGBM (30%) y LogisticRegression proxying Ridge (20%).
5. **Calibración**: `CalibratedClassifierCV` ajusta suavemente la curva de las probabilidades del ensamble usando Regresión Logística de Platt (`cv=3`).
6. **Inferencia**: El objeto interactivo `NBAPredictor` levanta el snapshot y los `.pkl` finales a memoria para respuestas < 50ms en producción vía FastAPI.

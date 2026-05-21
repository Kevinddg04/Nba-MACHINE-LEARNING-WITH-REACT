# 🏀 NBA ML Predictor

**Predictor de resultados NBA basado en Machine Learning** usando CatBoost con matchups simétricos para eliminar el sesgo de localía.

Desarrollado como proyecto académico de Desarrollo Web 2025.

**Autor**: Kevin Diaz

---

## ✨ Descripción

NBA ML Predictor utiliza un pipeline de Machine Learning entrenado con +50,000 partidos históricos de la NBA (2015–presente) para predecir:

- **Ganador** de un enfrentamiento entre dos equipos (CatBoostClassifier)
- **Puntaje total** esperado del partido (CatBoostRegressor)

El modelo emplea una estrategia de **matchups simétricos** (Anti-Home Bias) que genera filas espejo para cada partido, balanceando la importancia de la localía al 50/50. Esto evita que el modelo sobreestime la ventaja de jugar en casa.

### Características Principales

- 🔮 **Predictor**: Selecciona dos equipos y obtén la probabilidad de victoria
- 📊 **Standings**: Clasificación basada en estadísticas rolling (últimos 5 partidos)
- ⚔️ **Head-to-Head**: Simulación de 5 enfrentamientos directos
- 🧠 **Model Info**: Visualiza las features más importantes del modelo
- 📈 **Métricas**: Dashboard de hit rate y auditoría de predicciones

---

## 🚀 Quick Start

### Prerrequisitos

- Python 3.11+
- Node.js 18+ y npm
- (Opcional) Credenciales de Kaggle para actualizar datos

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/nba-ml-predictor.git
cd nba-ml-predictor
```

### 2. Backend — Instalar dependencias y entrenar modelos

```bash
cd backend
pip install -r requirements.txt

# Entrenar modelos (primera vez, ~5 minutos)
python ml_pipeline.py

# Iniciar servidor de desarrollo
python app.py
# → Servidor disponible en http://localhost:5000
```

### 3. Frontend — Instalar y correr

```bash
cd frontend
npm install

# Configurar URL del backend (crear archivo .env)
echo "VITE_API_URL=http://localhost:5000/api" > .env

npm run dev
# → Frontend disponible en http://localhost:5173
```

### 4. (Opcional) Actualizar datos desde Kaggle

```bash
# Configurar credenciales: ~/.kaggle/kaggle.json
cd backend
python kaggle_fetcher.py --force
python ml_pipeline.py  # Re-entrenar con datos actualizados
```

---

## 📁 Estructura del Proyecto

```
nba-project/
├── backend/
│   ├── app.py                 # API Flask (endpoints REST)
│   ├── ml_pipeline.py         # Pipeline ML: carga, features, entrenamiento
│   ├── kaggle_fetcher.py      # Descarga automática de datos desde Kaggle
│   ├── requirements.txt       # Dependencias de producción
│   ├── requirements-dev.txt   # Dependencias de desarrollo (pytest, black, etc.)
│   ├── Procfile               # Comando de inicio para Render
│   ├── runtime.txt            # Versión de Python para Render
│   ├── config/                # Configuración (logging)
│   ├── database/              # SQLAlchemy models + conexión
│   ├── metrics/               # Calculadora de hit rate
│   ├── ml/                    # Feature selection, ensemble, calibración
│   ├── tests/                 # Tests unitarios e integración (pytest)
│   └── models/                # Modelos entrenados (.pkl)
│       ├── classifier.pkl
│       ├── regressor.pkl
│       ├── classifier_features.pkl
│       ├── regressor_features.pkl
│       └── team_stats_snapshot.pkl
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Componente raíz + navegación
│   │   ├── api.ts             # Cliente HTTP para la API
│   │   ├── types.ts           # Tipos TypeScript
│   │   ├── index.css          # Estilos globales (dark theme NBA)
│   │   ├── pages/             # Predictor, Standings, HeadToHead, Dashboard
│   │   └── components/        # Componentes reutilizables
│   ├── package.json
│   └── vite.config.ts
├── TeamStatistics.csv          # Dataset (~24MB, ~50k filas)
├── render.yaml                 # Configuración de deployment Render
├── docker-compose.yml          # Stack local (PostgreSQL + Redis + Backend)
├── docs/
│   ├── API.md                  # Referencia completa de endpoints
│   ├── ARCHITECTURE.md         # Arquitectura del sistema
│   └── DEPLOYMENT.md           # Guía de despliegue
└── .github/workflows/
    └── test.yml                # CI/CD pipeline (pytest + black + pylint)
```

---

## ⚙️ Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `PORT` | Puerto del servidor | `5000` |
| `ADMIN_SECRET` | Secret para endpoint de actualización | `supersecreto` |
| `SELF_URL` | URL pública del backend (para keep-alive) | *(vacío)* |
| `KAGGLE_USERNAME` | Usuario de Kaggle API | *(desde kaggle.json)* |
| `KAGGLE_KEY` | Key de Kaggle API | *(desde kaggle.json)* |
| `DATABASE_URL` | URL de base de datos | `sqlite:///./nba.db` |
| `REDIS_URL` | URL de Redis | `redis://localhost:6379` |

---

## 🧪 Tests

```bash
cd backend

# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Correr todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=. --cov-report=term-missing

# Solo tests rápidos (sin entrenar modelos)
pytest tests/ -v -m "not slow"
```

---

## 🚢 Despliegue

El backend se despliega en **Render** y el frontend en cualquier servicio de archivos estáticos (Vercel, Netlify).

Ver [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) para instrucciones detalladas.

---

## 📚 Documentación

- [API Reference](docs/API.md) — Todos los endpoints con ejemplos curl
- [Architecture](docs/ARCHITECTURE.md) — Diseño del sistema y pipeline ML
- [Deployment Guide](docs/DEPLOYMENT.md) — Cómo desplegar en producción

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|------------|
| **ML** | CatBoost, scikit-learn, pandas, numpy |
| **Backend** | Flask (→ FastAPI), Gunicorn, SQLAlchemy |
| **Frontend** | React 19, TypeScript, Vite |
| **Datos** | Kaggle API, CSV (~50k filas) |
| **Deploy** | Render, GitHub Actions |
| **Testing** | pytest, Black, Pylint |

---

## 📄 Licencia

Proyecto académico — Universidad 2025.

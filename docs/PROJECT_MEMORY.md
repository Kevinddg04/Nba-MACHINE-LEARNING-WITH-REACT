# Memoria de Contexto: Evolución Arquitectónica del Proyecto NBA ML

Este documento sirve como registro oficial y memoria técnica de todas las mejoras estructurales, actualizaciones y subidas de nivel implementadas a lo largo de nuestro ciclo de desarrollo conjunto. Omite correcciones menores para enfocarse exclusivamente en las capacidades que elevaron el sistema a grado de producción.

## 1. Evolución de la Infraestructura y el Backend
* **Migración de Flask a FastAPI:** Reestructuramos el núcleo del servidor. Cambiamos de Flask a FastAPI para ganar tiempos de respuesta ultrarrápidos, asincronía nativa (`async/await`) y una validación de datos a prueba de balas impulsada por `Pydantic`.
* **Capa de Almacenamiento en Caché (Redis):** Se desarrolló un decorador inteligente de caché (`@cache_response`). Esto guarda temporalmente las respuestas pesadas de los Standings y Equipos para aligerar la carga de la API, incluyento una política de "Degradación Elegante" (El servidor sabe ignorar a Redis y seguir trabajando solo si este se apaga).
* **Blueprint de Despliegue (Infrastructure as Code):** Escribimos un archivo `render.yaml` puro. Esto transformó un proyecto manual en un despliegue Infra-Automático (Frontend sirviendo como sitio web estático ultra-rápido y el Backend de Python corriendo un WebService independiente).
* **El Motor Anti-Sueño (Keep-Alive):** Integramos un ciclo asíncrono infinito en el momento en el que el servidor nace (`startup`), que se hace auto-pings cada 14 minutos. Esto engaña y elude inteligentemente los modos de suspensión del plan gratuito de Render garantizando servicio 24/7.

## 2. Automatización y Autonomía (El "Loop Diario")
* **Extracción de Datos de Kaggle Autónoma:** Construimos `kaggle_fetcher.py`. Un sistema que se salta la necesidad manual humana. Usa un Súper Token (API de Kaggle) para descargar periódicamente la base de datos de la NBA más fresca del mundo, combinándola y purificándola contra nuestra base histórica (`TeamStatistics.csv`) sin romper ni duplicar registros.
* **Integración Continua Neural (24h Loop):** Insertamos un reloj perpetuo en el servidor (`daily_learning_loop`). Cada madrugada, sin intervención de nuestra parte, el servidor desata dos sub-procesos limpios (para salvaguardar su RAM limitada de 512MB): descarta información obsoleta, obliga a la Inteligencia Artificial a leer el nuevo CSV, calibra una vez más todos los parámetros estadísticos del Baloncesto actual y recarga la memoria del sistema en vivo para el Frontend.

## 3. Avanzando la Inteligencia Artificial (Machine Learning)
* **Arquitectura de Fusión (Ensemble Learning):** Trasladamos la carga a un sistema de doble vía fusionando CatBoost + LightGBM. Uno especializado en árboles precisos categóricos y el otro en rapidez en el gradiente espacial.
* **Calibración de Realidad Deportiva en H2H y Probabilidades:** Implementamos algoritmos probabilísticos de Gauss (`random.gauss`) al simular escenarios "Head to Head" (Enfrentamientos Directos). Esto dotó al cerebro de la IA para reconocer, respetar e integrar "Milagros Deportivos" (Sorpresas de Underdogs) evitando la tiranía irreal de resultados de victoria al azar limpios.
* **Ingeniería de Características en la Cancha:** Extraímos parámetros puros de "Feature Engineering". Programulamos métricas de inercia y dinámica en lugar de simple matemáticas. Creamos "Rachas de Ganancia" (Win Streaks) y "Rating Defensivo Continuo de los Últimos 10 Juegos", dándole un concepto de la "Forma y Moral Deportiva" al motor inteligente.

## 4. Legibilidad y Localización Universales
* **Transmutación Estructural a Español Nativo:** Un inmenso refactor donde pulimos los módulos completos (desde la predicción hasta la extracción) borrando los rastros de "Spanglish".
* **Refinamiento de la Auditoría Log:** Una base robusta de historial local (SQL `PredictionLog`) lista para guardar el récord de predicción y compararlo pragmáticamente con el margen de la vida real una vez las métricas lo demanden.

# Deployment Guide

El proyecto NBA ML Predictor ha sido migrado a un stack estandarizado de Docker y uvicorn, optimizado para plataformas PaaS como Render, Railway o tu VPS con `docker-compose`.

## 1. Despliegue en Render.com (Vía render.yaml)
El repositorio viene con un `render.yaml` ("Blueprint") configurado de caja.

### Pasos:
1. Sube tu código a GitHub.
2. Crea una cuenta en Render, ve a **Blueprints**, y apunta al repo.
3. Render levantará automáticamente el servicio web instalando las librerías necesarias.
4. **Consideración Redis**: Render no levanta docker-compose en servicios libres. El proyecto está construido para ser **Fault-Tolerant** a Redis. Si subes la aplicación sin vincular una base Redis, el caché simplemente será ignorado y servirá desde memoria viva sin arrojar HTTP 500s.

## 2. Despliegue en tu propia Servidor (Océano Digital / AWS)
Si tienes el binario `docker` y `docker-compose`:
```bash
# Entra a la ruta del proyecto
cd nba-project

# Orquesta y despliega
docker-compose up -d --build
```
Estaremos corriendo en el port 8000. `docker-compose` sí levantará la base Redis aislada unida eficientemente a la red interna del contenedor de nuestra API `backend`.

## 3. Actualización de Datos (Cron / Endpoint)
En el Front, existe la posibilidad de enviar una petición asíncrona autenticada por contraseña local:
```
POST /api/admin/update
{ "secret": "MI_LLAVE" }
```
Esto lanza una `BackgroundTask` que descarga un nuevo dataset desde Kaggle y recarga en memoria los pesos o el snapshot asíncronamente mientras tu servidor no deja de operar.

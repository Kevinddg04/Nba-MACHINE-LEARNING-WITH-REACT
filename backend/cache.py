import json
import os
import redis
from functools import wraps
from config.logging_config import get_logger

logger = get_logger(__name__)

# Redis Connection Setup
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Ping to test connection, timeout quickly if not available
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info(f"✅ Conectado a Redis en {REDIS_URL}")
except (redis.ConnectionError, redis.TimeoutError) as e:
    REDIS_AVAILABLE = False
    logger.warning(f"⚠️ Redis no disponible, funcionando en modo sin caché. Razón: {str(e)}")


def get_cache(key: str):
    """Obtiene un valor de la caché de manera segura."""
    if not REDIS_AVAILABLE:
        return None
    try:
        val = redis_client.get(key)
        if val:
            return json.loads(val)
        return None
    except Exception as e:
        logger.error(f"Error leyendo caché para {key}: {e}")
        return None


def set_cache(key: str, value: dict | list, expire_seconds: int = 3600):
    """Guarda un valor en la caché si está disponible."""
    if not REDIS_AVAILABLE:
        return
    try:
        redis_client.setex(key, expire_seconds, json.dumps(value))
    except Exception as e:
        logger.error(f"Error guardando caché para {key}: {e}")


def cache_response(expire_seconds: int = 3600):
    """
    Decorador para endpoints estáticos/lentos.
    Usa el nombre de la función y conectores como llave primaria.
    La cache es fault-tolerant.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key based on function name and args
            key_parts = [func.__name__]
            if kwargs:
                key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
            cache_key = ":".join(key_parts)
            
            # Intenta obtenerlo de cache
            cached = get_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache Hit: {cache_key}")
                return cached
                
            # Si no, computa la respuesta
            if getattr(func, '_is_coroutine', False) or bool(func.__code__.co_flags & 0x80):
                response = await func(*args, **kwargs)
            else:
                response = func(*args, **kwargs)
                
            # Guarda asíncronamente o sincrónicamente dependiendo del cliente (ahora es síncrono safe)
            set_cache(cache_key, response, expire_seconds)
            return response
            
        return wrapper
    return decorator

"""Configuración leída de variables de entorno (o de un fichero .env en local).

Todo lo que cambia entre entornos (local, CI, producción) vive aquí y no en el código:
así el mismo contenedor sirve para todos y los secretos nunca se suben a git.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://localhost/amr_agent"

    # Cualquier API compatible con OpenAI: Groq (por defecto), Gemini, Ollama en local...
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 30.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # 1 hilo: en Render Free hay 0.1 CPU y 512 MB; más hilos solo gastan RAM.
    embedding_threads: int = 1

    top_k: int = 5

    # NCBI E-utilities: la API key es opcional (sube el límite de 3 a 10 peticiones/s).
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

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
    # En la API solo se reintenta si el proveedor pide esperar poco: con más, se responde 503 ya.
    llm_max_retry_wait_s: float = 5.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # 1 hilo: en Render Free hay 0.1 CPU y 512 MB; más hilos solo gastan RAM.
    embedding_threads: int = 1

    top_k: int = 5
    # Búsqueda por palabras clave: se ignoran los términos presentes en más del 10 % de los
    # abstracts (ver retrieval.py y docs/decisions.md, decisión 14). 1.0 = no ignorar ninguno.
    keyword_max_df: float = 0.1
    # Peso de la lista de palabras clave en la fusión RRF (la vectorial pesa 1.0).
    keyword_weight: float = 1.0

    # Límites de /ask en la demo pública (0 = sin límite). Ver api/ratelimit.py.
    ask_per_ip_per_minute: int = 5
    ask_per_day: int = 50

    # Agente (/agent). Cada pregunta gasta ~8.000-12.000 tokens (3-4 veces un /ask), y la cuota
    # diaria de Groq (200.000 tokens) es compartida: 50 /ask (~150.000) + 15 /agent la llenan.
    agent_enabled: bool = True  # False = no arrancar amr-mcp (ahorra ~100 MB de RAM)
    # 1/min: una pregunta gasta ~11.700 tokens y Groq Free da 8.000/min, así que la segunda en el
    # mismo minuto fallaría con un 503 del proveedor. Mejor un 429 propio con cuenta atrás.
    agent_per_ip_per_minute: int = 1
    agent_per_day: int = 15
    agent_max_steps: int = 5
    agent_token_budget: int = 12_000
    mcp_timeout_s: float = 30.0

    # Modo literatura de /agent (pestaña "Papers" de la demo, decisión 37): busca en todo PubMed y
    # lee el texto completo de PMC. No usa amr-mcp ni la base de datos. Una pregunta gasta
    # ~13.000-16.000 tokens, el doble que el Groq Free deja por minuto (8.000).
    literature_enabled: bool = True
    literature_per_ip_per_minute: int = 1
    literature_per_day: int = 5  # ~80.000 de los 200.000 tokens diarios, compartidos con el resto
    literature_token_budget: int = 16_000
    # Aquí sí se espera a que pase el límite por minuto: sin esperar, casi toda pregunta acabaría
    # en 503 a mitad de camino. La pregunta tarda más (~1-2 min), pero termina.
    literature_max_retry_wait_s: float = 60.0

    # NCBI E-utilities: la API key es opcional (sube el límite de 3 a 10 peticiones/s).
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

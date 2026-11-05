# Imagen de producción (Render Free: 512 MB de RAM y 0,1 CPU).
#
# Dos etapas: la primera instala dependencias con uv y descarga el modelo de embeddings;
# la segunda copia solo el resultado, sin uv ni cachés, para que la imagen sea más pequeña.

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Primero solo las dependencias: esta capa se reutiliza mientras no cambie uv.lock,
# así que cambiar el código no obliga a reinstalarlo todo.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# El modelo (~70 MB) va dentro de la imagen. Render Free no tiene disco persistente: si se
# descargara al arrancar, cada arranque en frío tardaría más y dependería de Hugging Face.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"


FROM python:3.13-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

# Render indica el puerto en $PORT. `--proxy-headers` hace que request.client sea la IP real
# del visitante (Render va delante como proxy), que es la que usa el límite de peticiones.
CMD ["sh", "-c", "exec uvicorn amr_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]

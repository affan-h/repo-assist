# Stage 1: build the React UI.
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python app that serves the API and the built UI.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data /opt/models && chown app:app /data /opt/models

WORKDIR /app
# Dependencies first, in their own layer, so code changes don't re-download them.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
        > /tmp/requirements.txt \
    && pip install --no-cache-dir --default-timeout=120 --retries 10 -r /tmp/requirements.txt
COPY README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY --from=web /web/dist ./frontend/dist

ENV REPO_ASSIST_DATA_DIR=/data \
    REPO_ASSIST_MODEL_DIR=/opt/models \
    REPO_ASSIST_FRONTEND_DIR=/app/frontend/dist \
    PYTHONUNBUFFERED=1

USER app
# Bake the embedding model and the main tree-sitter grammars into the image so ingesting
# needs no downloads (other languages' grammars are fetched on first use).
RUN python -c "from repo_assist.embeddings import get_embedder; get_embedder()" \
    && python -c "import tree_sitter_language_pack as p; from repo_assist.ingest.code import SYMBOL_LANGUAGES as L; p.download(L)"

VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["repo-assist", "serve", "--host", "0.0.0.0", "--port", "8000"]

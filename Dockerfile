# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/* \
  && pip install uv --no-cache-dir

COPY requirements.txt .
# Install CPU-only torch first to avoid the 2GB CUDA wheel from PyPI.
# uv's resolver is 10-100x faster than pip and avoids backtracking on loose >= constraints.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY . .

RUN mkdir -p storage/pdfs chroma_db

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=docker
ENV CHROMA_DIR=/app/chroma_db
ENV STORAGE_DIR=/app/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

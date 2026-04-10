FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage/pdfs chroma_db

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=docker
ENV CHROMA_DIR=/app/chroma_db
ENV STORAGE_DIR=/app/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

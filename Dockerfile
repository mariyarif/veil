FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TESSERACT_CMD=/usr/bin/tesseract
CMD gunicorn -w 1 --threads 2 --timeout 180 -b 0.0.0.0:${PORT:-10000} app:app
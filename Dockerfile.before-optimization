FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/task_3_5_champion.keras \
    DATABASE_PATH=/app/instance/visionguard.db \
    MAX_UPLOAD_MB=5 \
    THRESHOLD=0.4980708956718445

WORKDIR /app

# Install Python dependencies first for better Docker layer caching.
COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy only files required to run the application.
COPY app/ ./app/
COPY run.py .
COPY models/task_3_5_champion.keras ./models/task_3_5_champion.keras

# Create a non-root application user and writable directories.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/instance \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Check whether the Flask application is responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)" || exit 1

CMD ["waitress-serve", "--listen=0.0.0.0:8000", "run:app"]

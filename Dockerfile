FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCWISE_STORAGE_DIR=/app/storage

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --shell /usr/sbin/nologin docwise \
    && mkdir -p /app/storage \
    && chown -R docwise:docwise /app/storage

EXPOSE 8501
VOLUME ["/app/storage"]
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
USER docwise
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

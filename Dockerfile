FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

COPY --chown=appuser:appuser app.py inference.py ./
COPY --chown=appuser:appuser templates templates
COPY --chown=appuser:appuser static static

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"

CMD ["python", "app.py"]

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY service /app/service
COPY metrics_adapter /app/metrics_adapter
USER 10001
EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

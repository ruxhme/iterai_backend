FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements.txt
# Force CPU-only PyTorch to avoid huge CUDA image layers on Railway.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install -r requirements.txt && \
    pip cache purge

COPY main.py title_engine.py database.py seed_vectors.py ./

EXPOSE 8000

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"

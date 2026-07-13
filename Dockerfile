FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    TORCH_NUM_THREADS=1 \
    PORT=7860

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && pip install --no-cache-dir --no-deps grad-cam==1.5.5

COPY server ./server
COPY web ./web
COPY models ./models

EXPOSE 7860

CMD ["sh", "-c", "cd server && python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

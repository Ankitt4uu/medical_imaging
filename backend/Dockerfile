FROM python:3.11-slim
WORKDIR /app
COPY requirements/ requirements/
# CPU-only torch keeps the image small; swap the index URL for a CUDA build on GPU hosts
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements/backend.txt
COPY medimg medimg
COPY backend backend
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
RUN useradd -m app && mkdir -p /app/state && chown app /app/state
USER app
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

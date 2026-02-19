FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (smaller image, no GPU on VPS)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install remaining deps (skip torch since already installed)
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null; \
    pip install --no-cache-dir joblib 2>/dev/null; \
    exit 0

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

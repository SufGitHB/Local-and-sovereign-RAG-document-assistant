FROM python:3.11-slim

WORKDIR /app

COPY rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

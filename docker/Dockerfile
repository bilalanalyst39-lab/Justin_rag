FROM python:3.11-slim

WORKDIR /app

COPY ../backend-requirements.txt .
RUN pip install --no-cache-dir -r backend-requirements.txt

COPY ../backend_1.py .

EXPOSE 8000

CMD ["uvicorn", "backend_1:app", "--host", "0.0.0.0", "--port", "8000"]

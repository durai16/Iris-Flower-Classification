FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 is required so the API is reachable from outside the container.
# 127.0.0.1 would listen only inside the container itself.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
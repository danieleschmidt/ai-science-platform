FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY *.py ./
COPY config/ config/

RUN mkdir -p logs data backups

EXPOSE 8000

CMD ["python3", "-c", "print('AI Science Platform ready')"]

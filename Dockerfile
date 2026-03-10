FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 10000

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000"]
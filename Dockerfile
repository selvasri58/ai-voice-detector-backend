FROM python:3.10-slim

# Force Python to print logs immediately to the Render dashboard
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ffmpeg (required for audio conversion)
RUN apt-get update && apt-get install -y ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files 
COPY . .

# Expose port
EXPOSE 10000

# Start server with a 120s timeout and visible error logging
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "debug", "server:app"]
FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg (required for audio conversion)
RUN apt-get update && apt-get install -y ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (important if you add more later)
COPY . .

# Expose port
EXPOSE 10000

# Start server
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app"]
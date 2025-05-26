FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install system dependencies needed for curl and gunicorn
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama CLI using official install script

# Expose Flask port and Ollama API port
EXPOSE 5000 

# Version simple avec timeout long
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "1800", "--keep-alive", "300", "app:app"]
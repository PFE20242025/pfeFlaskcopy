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
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull phi3 model into Ollama
RUN ollama run phi3

# Expose Flask port and Ollama API port
EXPOSE 5000 11434

# Start Ollama server in background, then launch Flask app with gunicorn
CMD ollama serve & gunicorn --bind 0.0.0.0:5000 app:app

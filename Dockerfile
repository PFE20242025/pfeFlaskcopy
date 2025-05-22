# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all files (app code, requirements, etc)
COPY . .

# Install system dependencies needed for Ollama CLI and unzip
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and install Ollama CLI
RUN curl -L https://ollama.com/download/ollama-linux-amd64.zip -o /tmp/ollama.zip \
    && unzip /tmp/ollama.zip -d /usr/local/bin/ \
    && rm /tmp/ollama.zip

# Verify Ollama installation
RUN ollama --version

# Pull phi3 model for Ollama usage
RUN ollama pull phi3

# Expose Flask port and Ollama API port
EXPOSE 5000 11434

# Run Ollama server in background and then start Flask app
CMD ollama serve & gunicorn --bind 0.0.0.0:5000 app:app

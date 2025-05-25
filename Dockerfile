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

# Only expose Flask port - Ollama runs locally inside container
EXPOSE 5000

# Start Gunicorn first, then Ollama in background on localhost
CMD sh -c "gunicorn --bind 0.0.0.0:5000 app:app & sleep 5 && ollama serve --host 127.0.0.1 & sleep 10 && ollama pull phi3 && wait"
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

# Set Ollama host and expose necessary ports
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 5000 11434

# Start Ollama server and pull models once, then start Flask app
CMD sh -c "ollama serve & \
    sleep 10 && \
    echo 'Pulling models on first container start...'; \
    ollama pull phi3 && \
    ollama pull model2 && \
    ollama pull model3 && \
    echo 'All models pulled successfully, starting Flask app with Gunicorn...'; \
    gunicorn --bind 0.0.0.0:5000 --timeout 120 --workers 9 app:app"
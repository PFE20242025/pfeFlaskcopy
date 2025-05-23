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

# Start Ollama server in the background and ensure the model is pulled before running the app
CMD sh -c "ollama serve & \
    sleep 10 && \
    if [ ! -f /app/.models_pulled ]; then \
        echo 'Pulling all models for the first time...'; \
        ollama pull phi3 && \
        ollama pull model2 && \
        ollama pull model3 && \
        touch /app/.models_pulled; \
    fi && \
    echo 'Models are ready, starting Flask app with Gunicorn...'; \
    gunicorn --bind 0.0.0.0:5000 --timeout 120 --workers 9 app:app"

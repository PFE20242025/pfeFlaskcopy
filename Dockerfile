FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl procps && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
# Expose ports
EXPOSE 5000 11434

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Ollama server..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
echo "Waiting for Ollama to be ready..."\n\
for i in {1..30}; do\n\
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then\n\
        echo "Ollama is ready!"\n\
        break\n\
    fi\n\
    echo "Waiting for Ollama... ($i/30)"\n\
    sleep 5\n\
done\n\
\n\
echo "Pulling phi3 model..."\n\
ollama pull phi3 || echo "Failed to pull phi3 model"\n\
\n\
echo "Starting Flask application on port 5000..."\n\
exec gunicorn --bind 0.0.0.0:5000 --timeout 300 --workers 1 --worker-class sync app:app\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
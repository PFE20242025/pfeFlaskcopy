FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into container
COPY . .

# Install Ollama CLI using official install script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create supervisor configuration
RUN mkdir -p /var/log/supervisor

# Create supervisor config file
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisor/supervisord.pid

[program:ollama]
command=ollama serve
user=root
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/ollama_err.log
stdout_logfile=/var/log/supervisor/ollama_out.log
environment=OLLAMA_HOST="0.0.0.0:11434"

[program:model_puller]
command=/app/pull_models.sh
user=root
autostart=true
autorestart=false
startsecs=0
stderr_logfile=/var/log/supervisor/model_puller_err.log
stdout_logfile=/var/log/supervisor/model_puller_out.log

[program:flask]
command=gunicorn --bind 0.0.0.0:5000 --timeout 120 --workers 4 --log-level debug app:app
directory=/app
user=root
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/flask_err.log
stdout_logfile=/var/log/supervisor/flask_out.log
depends_on=ollama
EOF

# Create model pulling script
COPY <<EOF /app/pull_models.sh
#!/bin/bash
set -e

echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama is ready. Checking models..."

if [ ! -f /app/.models_pulled ]; then
    echo "Pulling models for the first time..."
    ollama pull phi3
    # Add other models as needed
    # ollama pull model2
    # ollama pull model3
    touch /app/.models_pulled
    echo "Models pulled successfully"
else
    echo "Models already pulled"
fi

echo "Model puller finished"
# Keep the process alive briefly to satisfy supervisor
sleep 5
EOF

# Make script executable
RUN chmod +x /app/pull_models.sh

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

# Expose necessary ports
EXPOSE 5000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use supervisor to manage all processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
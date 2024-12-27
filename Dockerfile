FROM python:3.11-slim

WORKDIR /app
ENV POETRY_HOME=/opt/poetry
ENV ENVIRONMENT=cloud
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry

# Cache-Busting: Install dependencies first
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main \
    && rm -rf ~/.cache/pypoetry

# Copy application code at the end
COPY . .

# Clear any potential Python cache
RUN find . -type d -name __pycache__ -exec rm -r {} + || true

# Make the script executable
RUN chmod +x run_services.sh

EXPOSE 8080 8081
CMD ["./run_services.sh"]
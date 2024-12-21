FROM python:3.11-slim

WORKDIR /app
ENV POETRY_HOME=/opt/poetry
ENV ENVIRONMENT=cloud

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# API Dependencies
RUN pip install fastapi uvicorn

COPY . .

# Make the script executable
RUN chmod +x run_services.sh

EXPOSE 8080 8081

CMD ["./run_services.sh"]
# Python 3.11 statt 3.9 verwenden
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Poetry installieren
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry

# Poetry Files kopieren und Dependencies installieren
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# App-Code kopieren
COPY . .

EXPOSE 8080

CMD streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0
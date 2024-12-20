FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Poetry installieren
RUN curl -sSL https://install.python-poetry.org | python3 -

# Kopiere Poetry-Files
COPY pyproject.toml poetry.lock ./

# Poetry konfigurieren und Dependencies installieren
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# App-Code kopieren
COPY . .

EXPOSE 8080

# Streamlit starten
CMD streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0
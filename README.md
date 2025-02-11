# Analogierechner: Semantische Analyse

Eine Streamlit-App zur Berechnung von Wort-Analogien in Deutsch und Englisch. Die App nutzt Word Embeddings (FastText für Deutsch, GloVe für Englisch), um semantische Beziehungen zwischen Wörtern zu analysieren.

## Features
- Berechnung von Wort-Analogien (z.B. "Deutschland : Berlin = Frankreich : Paris")
- Unterstützung für deutsche und englische Sprache
- Ähnliche Wörter finden
- Interaktive Visualisierung der Ergebnisse

## Installation & lokales Setup

### Voraussetzungen
- Python 3.9+
- Poetry für Dependency Management

### Setup
1. Repository klonen:
```bash
git clone [repository-url]
cd analogierechner
```

2. Dependencies installieren:
```bash
poetry shell
poetry install
pip install -e .
```

3. Modell-Dateien herunterladen:
- FastText Deutsch: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz
- GloVe English: https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation?resource=download

4. Modell-Dateien entpacken und in `data/` Verzeichnis platzieren:
```
data/
  ├── cc.de.300.bin
  └── glove.6B.100d.txt
```

5. App starten 
```bash
poetry run streamlit run src/streamlit_app.py
```


6. Tests ausführen
# Run all tests
python -m pytest

# Run tests with output
python -m pytest -v

# Run specific test file
python -m pytest test/test_embedding_handler.py -v

# Run tests with print output
python -m pytest -s

## Google Cloud Deployment

### Voraussetzungen
- Google Cloud Account
- Google Cloud CLI installiert
- Billing aktiviert

### Deployment-Schritte

1. Google Cloud CLI installieren & einrichten:
```bash
# Für Mac
brew install google-cloud-sdk
gcloud auth login
```

2. Projekt erstellen/auswählen:
```bash
gcloud projects create [PROJECT-ID]  # Optional, falls noch kein Projekt existiert
gcloud config set project [PROJECT-ID]
```

3. APIs aktivieren:
```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com
```

4. Service Account einrichten:
```bash
# Service Account erstellen
gcloud iam service-accounts create cloudbuild-deployer \
    --description="Cloud Build Deployer" \
    --display-name="Cloud Build Deployer"

# Rollen zuweisen
export PROJECT_ID=[PROJECT-ID]
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/cloudbuild.builds.builder"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/run.admin"
```

5. Storage Bucket für Modelle erstellen & Dateien hochladen:
```bash
# Bucket erstellen
gsutil mb -l europe-west3 gs://${PROJECT_ID}_models

# Modell-Dateien hochladen
gsutil cp data/cc.de.300.bin gs://${PROJECT_ID}_models/data/
gsutil cp data/glove.6B.100d.txt gs://${PROJECT_ID}_models/data/
gsutil cp data/de_50k_most_frequent.txt gs://${PROJECT_ID}_models/data/
gsutil cp data/en_50k_most_frequent.txt gs://${PROJECT_ID}_models/data/
```

6. App und API deployen
```bash
gcloud auth login

gcloud run deploy analogierechner \
  --source . \
  --platform managed \
  --region europe-west3 \
  --allow-unauthenticated \
  --service-account="cloudbuild-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
```

# lokal
```bash
## Entwicklung mit Logs
docker-compose up --build

## Entwicklung im Hintergrund
docker-compose up -d --build

=> danach: http://localhost:8080

## um den memory usage zu sehen:
docker stats

## Logs anschauen
docker-compose logs -f

## nach änderungen des python codes
docker-compose restart

## Container stoppen
docker-compose down
```

### Kosten & Monitoring

- Billing Alarm einrichten:
  1. Google Cloud Console → Billing → Budgets & Alerts
  2. "CREATE BUDGET" wählen
  3. Budget und Alarmschwellen festlegen

- Typische Kosten:
  - Cloud Run: Free Tier mit 2 Millionen Requests/Monat
  - Storage: Free Tier mit 5GB/Monat
  - Geschätzte Kosten bei mittlerer Nutzung: $5-15/Monat

## Projektstruktur
```
analogierechner/
├── data/                    # Modell-Dateien
├── logs/                    
├── src/
│   ├── core/               # Core Funktionalität
│   │   └── embedding_handler.py
├── streamlit_app.py        # Hauptanwendung
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Technologie-Stack
- Streamlit für das Frontend
- FastText & GloVe für Word Embeddings
- Poetry für Dependency Management
- Google Cloud Storage für Modell-Speicherung
- Google Cloud Deploy für Deployment
- Google Cloud Run für Hosting


# source of word list
## ger 5k
https://github.com/frekwencja/most-common-words-multilingual/blob/main/data/wordfrequency.info/de.txt

## ger 50k
https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/de/de_50k.txt

## en 50k
https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/en/en_50k.txt


# source of embeddings
## en glove 100d
https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation?resource=download


#!/bin/bash

# Dein Google Cloud-Projekt und Repository
PROJECT_ID="dein-projekt-id"
REPO="analogierechner"
REGION="europe-west3"  # Falls dein Registry-Standort anders ist, ändere es

# Authentifiziere Google Cloud
gcloud auth configure-docker $REGION-docker.pkg.dev

# Liste alle Images und speichere sie
IMAGES=$(gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/$REPO --format="value(package)")

# Gehe durch jede Image-Name und lösche alte Versionen
for IMAGE in $IMAGES; do
    echo "Überprüfe: $IMAGE"

    # Alle Tags/Verschiedene Versionen des Images abrufen
    TAGS=$(gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE --format="value(version)" --sort-by="~updateTime")

    # Neueste Version behalten
    NEWEST=$(echo "$TAGS" | head -n 1)
    echo "Neueste Version bleibt erhalten: $NEWEST"

    # Ältere Versionen löschen
    for OLD_TAG in $(echo "$TAGS" | tail -n +2); do
        echo "Lösche: $IMAGE:$OLD_TAG"
        gcloud artifacts docker images delete $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE@$OLD_TAG --delete-tags --quiet
    done
done

echo "✅ Alle alten Artefakte wurden gelöscht!"

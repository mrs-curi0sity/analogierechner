import os
import fasttext
import fasttext.util

def download_german_model(output_dir='data'):
    """
    Lädt das deutsche FastText-Modell herunter.
    Speichert es im data-Verzeichnis des Projekts.
    """
    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Vollständiger Pfad zur Modelldatei
    model_path = os.path.join(output_dir, 'cc.de.300.bin')
    
    # Nur herunterladen, wenn die Datei noch nicht existiert
    if not os.path.exists(model_path):
        # Deutschen FastText-Modell herunterladen
        fasttext.util.download_model('de', if_exists='ignore')
        
        print(f"Deutsches FastText-Modell heruntergeladen nach {model_path}")
    else:
        print(f"Modell existiert bereits unter {model_path}")

if __name__ == "__main__":
    download_german_model()
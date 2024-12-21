import pandas as pd
import requests
from typing import List
from time import sleep

def process_csv(input_path: str, output_path: str, api_url: str, batch_size: int = 10):
    """
    Liest CSV, sendet Anfragen an API und speichert Ergebnisse
    
    CSV Format sollte sein:
    language,word1,word2,word3
    en,Germany,Berlin,France
    de,Mann,König,Frau,
    ...
    """
    # CSV lesen
    df = pd.read_csv(input_path)
    
    # Daten in Batches verarbeiten
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Batch in Request-Format umwandeln
        requests_data = [
            {
                "word1": row.word1,
                "word2": row.word2,
                "word3": row.word3,
                "language": row.language
            }
            for _, row in batch.iterrows()
        ]
        
        # API-Request senden
        try:
            response = requests.post(f"{api_url}/batch-analogy", json=requests_data)
            response.raise_for_status()
            batch_results = response.json()["results"]
            results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
        except Exception as e:
            print(f"Error processing batch: {e}")
            results.extend([None] * len(batch))
        
        # Kurze Pause zwischen Batches
        sleep(0.1)
    
    # Ergebnisse zur DataFrame hinzufügen
    df['result'] = results
    
    # Ergebnisse speichern
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Beispiel-Nutzung
    INPUT_PATH = "data/analogies.csv"
    OUTPUT_PATH = "data/analogies_results.csv"
    API_URL = "https://analogierechner-762862809820.europe-west3.run.app"  # Anpassen!
    
    process_csv(INPUT_PATH, OUTPUT_PATH, API_URL)
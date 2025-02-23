import os
import pandas as pd
import requests
from typing import List
from time import sleep


# Umgebungsvariablen
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

# Base Paths
# Base Paths
BASE_PATH = (
    "gs://analogierechner-models/data" 
    if ENVIRONMENT == "cloud" 
    else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)

API_URL = "https://analogierechner-762862809820.europe-west3.run.app" if ENVIRONMENT == "cloud" else "http://localhost:8081"

# Input/Output Paths
INPUT_PATH = os.path.join(BASE_PATH, "analogies_2025_02_21.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "analogies_2025_02_21_results.csv")

def process_csv(input_path: str, output_path: str, api_url: str, batch_size: int = 5):
    df = pd.read_csv(input_path)
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        requests_data = [
            {
                "word1": row.word1,
                "word2": row.word2,
                "word3": row.word3,
                "language": row.language
            }
            for _, row in batch.iterrows()
        ]
        
        print(f"\nSending batch request {i//batch_size + 1}/{len(df)//batch_size + 1}")
        try:
            response = requests.post(
                f"{api_url}/batch-analogy", 
                json=requests_data,
                timeout=60,  # Längerer Timeout
                headers={'Content-Type': 'application/json'}
            )
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                batch_results = response.json()["results"]
                results.extend(batch_results)
            else:
                print(f"Error response: {response.text}")
                results.extend([None] * len(batch))
        except requests.Timeout:
            print("Request timed out")
            results.extend([None] * len(batch))
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.text if e.response else 'No response'}")
            results.extend([None] * len(batch))
        
    df['result'] = results
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    print(f"Running in {ENVIRONMENT} environment")
    print(f"Using API: {API_URL}")
    print(f"Reading from: {INPUT_PATH}")
    print(f"Writing to: {OUTPUT_PATH}")
    
    process_csv(INPUT_PATH, OUTPUT_PATH, API_URL)
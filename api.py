fimport csv
from datetime import datetime
import os

# CSV Logger Setup
LOG_FILE = 'analogies_log.csv'
CSV_HEADERS = ['timestamp', 'language', 'word1', 'word2', 'word3', 'output']

# Datei erstellen, falls sie nicht existiert
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

def log_to_csv(language, word1, word2, word3, output):
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            language,
            word1,
            word2,
            word3,
            output
        ])

@app.post("/analogy")
async def get_analogy(request: AnalogyRequest):
    handler = EmbeddingHandler(language=request.language)
    results, _ = handler.find_analogy(request.word1, request.word2, request.word3, "")
    result = results[0][0] if results else None
    
    log_to_csv(
        request.language,
        request.word1,
        request.word2,
        request.word3,
        result
    )
    
    return {"results": result}

@app.post("/batch-analogy")
async def batch_analogy(requests: list[AnalogyRequest]):
    results = []
    handlers = {}
    
    for request in requests:
        if request.language not in handlers:
            handlers[request.language] = EmbeddingHandler(request.language)
        
        result, _ = handlers[request.language].find_analogy(
            request.word1, request.word2, request.word3, ""
        )
        results.append(result[0][0] if result else None)
    
    return {"results": results}
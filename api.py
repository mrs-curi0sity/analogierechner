from fastapi import FastAPI
from pydantic import BaseModel
from src.core.embedding_handler import EmbeddingHandler

app = FastAPI()

class AnalogyRequest(BaseModel):
    word1: str
    word2: str
    word3: str
    language: str = "en"

@app.post("/analogy")
async def get_analogy(request: AnalogyRequest):
    handler = EmbeddingHandler(language=request.language)
    results, _ = handler.find_analogy(request.word1, request.word2, request.word3, "")
    return {"results": results[0][0] if results else None}  # Nur bestes Ergebnis

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
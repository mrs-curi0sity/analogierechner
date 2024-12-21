# api.py
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
    return {"results": results}

# Beispiel-Nutzung mit Python requests:
import requests

data = [
    {"word1": "Germany", "word2": "Berlin", "word3": "France", "language": "en"},
    {"word1": "king", "word2": "queen", "word3": "man", "language": "en"},
    # ... mehr Beispiele
]

results = [requests.post("http://your-url/analogy", json=item).json() for item in data]
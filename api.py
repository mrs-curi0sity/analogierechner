from fastapi import FastAPI
from pydantic import BaseModel
from src.core.embedding_handler import EmbeddingHandler
from src.core.logger import logger

app = FastAPI()

class AnalogyRequest(BaseModel):
   word1: str
   word2: str 
   word3: str
   language: str = "de"

@app.post("/analogy")
async def get_analogy(request: AnalogyRequest):
   handler = EmbeddingHandler(language=request.language)
   results, _ = handler.find_analogy(request.word1, request.word2, request.word3, "")
   result = results[0][0] if results else None
   
   logger.log(
       'api',
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
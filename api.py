from fastapi import FastAPI
from pydantic import BaseModel
from src.core.embedding_handler import EmbeddingHandler
from src.core.logger import logger
from typing import List 

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
async def batch_analogy(requests: List[AnalogyRequest]):
    logger.info(f"Received batch request with {len(requests)} items")
    results = []
    handlers = {}
    
    for i, request in enumerate(requests):
        logger.info(f"Processing request {i+1}/{len(requests)}: {request}")
        try:
            if request.language not in handlers:
                logger.info(f"Initializing handler for language: {request.language}")
                handlers[request.language] = EmbeddingHandler(request.language)
            
            result, _, _ = handlers[request.language].find_analogy(
                request.word1, request.word2, request.word3, ""
            )
            results.append(result[0][0] if result else None)
            logger.info(f"Successfully processed request {i+1}")
        except Exception as e:
            logger.error(f"Error processing request {i+1}: {str(e)}")
            results.append(None)
    
    logger.info("Completed batch processing")
    return {"results": results}
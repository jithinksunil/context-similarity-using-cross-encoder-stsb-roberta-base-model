from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("FastAPI app started")

# Load the model once when the app starts
model = CrossEncoder("cross-encoder/stsb-roberta-base")

app = FastAPI()

# Request model
class SimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/similarity")
async def get_similarity(data: SimilarityRequest):
    score = model.predict([(data.sentence1, data.sentence2)])
    return {"similarity": float(score[0])}

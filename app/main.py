# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import predict_scores

app = FastAPI(title="Comment Toxicity API", version="1.0")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    scores = predict_scores(req.text)
    toxic_flag = scores["toxic"] >= 0.5
    return {
        "toxic": toxic_flag,
        "scores": scores
    }

from fastapi import FastAPI, Request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

MODEL_PATH = "saved_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = model.config.id2label


def predict_intent(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    req = await request.json()
    user_text = req["queryResult"]["queryText"]
    predicted_intent = predict_intent(user_text)
    return {
        "fulfillmentText": f"Detected intent: {predicted_intent}"
    }

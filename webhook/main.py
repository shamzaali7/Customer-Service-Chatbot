from fastapi import FastAPI, Request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()
tokenizer = BertTokenizer.from_pretrained("./saved_model")
model = BertForSequenceClassification.from_pretrained("./saved_model")
model.eval()

id2label = model.config.id2label

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

@app.post("/webhook")
async def webhook(request: Request):
    req = await request.json()
    user_text = req["queryResult"]["queryText"]

    predicted_intent = predict_intent(user_text)

    return {
        "fulfillmentText": f"I detected your intent as: {predicted_intent}"
    }

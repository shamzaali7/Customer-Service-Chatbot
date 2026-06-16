# Customer Service Chatbot — NLP Final Project

A BERT-based intent classification system for customer service automation. Fine-tuned on ~29,000 labeled utterances across 28 intents, served via a FastAPI webhook compatible with Dialogflow.

**Model on HuggingFace Hub:** [Hamzaali7/customer-service-chatbot](https://huggingface.co/Hamzaali7/customer-service-chatbot)

---

## Overview

This project fine-tunes `bert-base-uncased` to classify customer service queries into intents (e.g., `cancel_order`, `track_order`, `payment_issue`). The trained model is hosted on HuggingFace Hub and loaded at runtime by the webhook server.

**Key stats:**
- Training data: ~29,000 utterances, 28 intent classes
- Model: BERT (bert-base-uncased) fine-tuned with HuggingFace Transformers
- Deployment target: Heroku (CPU, Linux)
- Webhook framework: FastAPI + Uvicorn

---

## Project Structure

```
.
├── data/                          # Training CSV datasets
│   ├── 20000-Utterances-Training-dataset-...csv
│   └── Bitext_Sample_Customer_Service_Training_Dataset.csv
├── notebook/
│   └── Final_Project_NLP_Customer_Service_Chatbot.ipynb  # Training notebook
├── main.py                        # FastAPI webhook server
├── requirements.txt               # Deployment dependencies
├── Procfile                       # Heroku process definition
└── runtime.txt                    # Heroku Python version
```

The trained model is not stored in this repository — it is loaded automatically from HuggingFace Hub ([Hamzaali7/customer-service-chatbot](https://huggingface.co/Hamzaali7/customer-service-chatbot)) at startup.

---

## Setup

### Prerequisites

- Python 3.12

### Clone and install

```bash
git clone git@github.com:shamzaali7/Customer-Service-Chatbot.git
cd Customer-Service-Chatbot

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run locally

```bash
uvicorn main:app --reload
```

The model (~418 MB) downloads automatically from HuggingFace Hub on first run and is cached locally.

The webhook is available at `http://localhost:8000/webhook`.
A health check is available at `http://localhost:8000/health`.

---

## Training

Open the notebook and run all cells:

```bash
cd notebook
jupyter lab Final_Project_NLP_Customer_Service_Chatbot.ipynb
```

If a `saved_model/` directory exists at the project root, the notebook loads it and skips training. Otherwise it trains from scratch (~3 epochs on the full dataset) and saves the model to `saved_model/`.

**Additional dependencies for training:**

```
datasets
scikit-learn
rapidfuzz
nltk
matplotlib
accelerate
jupyterlab
jupyterlab_widgets
```

---

## API Reference

### `POST /webhook`

Accepts a Dialogflow fulfillment request and returns the predicted intent.

**Request body** (Dialogflow format):
```json
{
  "queryResult": {
    "queryText": "I want to cancel my order"
  }
}
```

**Response:**
```json
{
  "fulfillmentText": "Detected intent: cancel_order"
}
```

### `GET /health`

Returns `{"status": "ok"}` — useful for uptime monitoring.

---

## Deployment (Heroku)

This project is configured for Heroku with a CPU-only PyTorch build.

```bash
heroku create your-app-name
heroku buildpacks:set heroku/python
git push heroku main
```

`runtime.txt` pins Python 3.12. The `Procfile` starts Uvicorn on Heroku's dynamic `$PORT`. The model is downloaded from HuggingFace Hub on dyno startup.

> `requirements.txt` uses `torch==2.9.1+cpu` with PyTorch's CPU wheel index — this keeps the slug size manageable and avoids GPU dependencies on Heroku.

---

## Data Sources

Both datasets are from [Bitext](https://www.bitext.com/) and cover common e-commerce / customer service intent categories including order management, payment, shipping, returns, account issues, and product queries.

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Model | BERT (`bert-base-uncased`) via HuggingFace Transformers |
| Training | HuggingFace `Trainer` API |
| Inference server | FastAPI + Uvicorn |
| Deployment | Heroku |
| Model hosting | HuggingFace Hub |

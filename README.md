# Aspect-Based Sentiment Analysis for Nepali YouTube Comments

A full-stack NLP pipeline that extracts YouTube comments, identifies language, translates, detects targets, and predicts aspect-based sentiment along with a React frontend and interactive charts.

---

## Overview

This project performs **Target-Aspect-Based Sentiment Analysis (TABSA)** on Nepali YouTube comments. Given a YouTube URL, the system:

1. Extracts comments using the YouTube Data API
2. Identifies the language of each comment (Nepali Devanagari, English, Roman Nepali, Code-Mixed)
3. Translates English comments to Nepali Devanagari using IndicTrans2
4. Identifies named targets (politicians, parties, institutions, policies) using a fine-tuned XLM-RoBERTa NER model
5. Predicts the aspect and sentiment for each target
6. Displays results in a clean React frontend with sentiment and aspect charts

---

## Architecture

```
YouTube URL
    ↓
Comment Extractor (YouTube Data API)
    ↓
Language Identifier (FastText)
    ↓
    ├── NE_DEV → use directly
    ├── EN     → IndicTrans2 transliteration → Devanagari
    ├── NE_ROM → skip
    └── CODE_MIXED → skip
    ↓
Target Identifier (XLM-RoBERTa NER)
    ↓
Aspect + Sentiment Predictor (Keras + FastText embeddings)
    ↓
React Frontend (charts + result cards)
```

---

## Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| Language Identifier | FastText | Detects NE_DEV, NE_ROM, EN, CODE_MIXED |
| Translation | IndicTrans2 (AI4Bharat) | English → Nepali Devanagari |
| Target Identification | XLM-RoBERTa (fine-tuned NER) | Extracts POLITICIAN, PARTY, POLICY, INSTITUTION |
| Sentiment + Aspect | Keras CNN + BiLSTM + FastText (cc.ne.300.bin) | Predicts aspect and sentiment per target |

---

## Project Structure

```
NLP-project/
├── ml/
│   ├── base.py
│   ├── language_identifier.py
│   ├── translation_model.py
│   ├── target_model.py
│   └── devanagari.py
├── models/
│   ├── language_identifier/
│   ├── translation_model/
│   ├── target_model/
│   └── devanagari_models/
├── embeddings/
│   └── cc.ne.300.bin
├── notebooks/
│   ├── devanagari.ipynb
│   ├── translation.ipynb
│   ├── target_identification.ipynb
│   └── language_identification.ipynb
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── ResultCard.jsx
│       │   └── Charts.jsx
│       └── index.css
├── main.py
├── registry.py
├── pipeline.py
├── comment_extractor.py
├── requirements.txt
└── .env

```

---

## Setup

### Prerequisites

- Python 3.11 (64-bit) — [download](https://python.org/downloads)
- Node.js LTS — [download](https://nodejs.org)
- Git — [download](https://git-scm.com)

### 1. Clone the repository

```bash
git clone https://github.com/Nirika-Lamichhane/NLP-project.git
cd NLP-project
```

### 2. Create virtual environment

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Python dependencies

```bash
pip install fastapi==0.135.1 uvicorn==0.42.0 tensorflow==2.20.0 torch==2.10.0 "transformers>=4.51" fasttext-wheel==0.9.2 scikit-learn==1.8.0 numpy==1.26.4 pandas==2.3.3 google-api-python-client==2.192.0 python-dotenv==1.2.2 indictranstoolkit==1.1.1 indic-nlp-library-itt==0.1.1 accelerate seqeval sentencepiece==0.1.99 sacremoses==0.1.1 safetensors==0.7.0 pydantic==2.12.5 h5py==3.16.0 huggingface_hub==0.36.2 tokenizers==0.21.1 requests==2.32.5 httpx==0.28.1 gensim==4.4.0
```

### 4. Download model weights

Download the following from Google Drive and place them in the correct folders:

```
models/
  language_identifier/
    fasttext_language_model.bin
  transliteration_model/
    (all HuggingFace model files)
  target_model/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    label_mappings.json
    generic_targets.txt
  devanagari_models/
    devanagari_model.h5
    aspect_encoder.pkl
    sentiment_encoder.pkl
embeddings/
  cc.ne.300.bin
```

### 5. Create .env file

Create a `.env` file in the project root:

```
YOUTUBE_API_KEYS=your_youtube_api_key_here
```

Get a free YouTube Data API key from [Google Cloud Console](https://console.cloud.google.com).

### 6. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

### Terminal 1 — Start backend

```bash
.\venv\Scripts\activate
uvicorn main:app --port 8000
```

Wait for all models to load (3-5 minutes):

```
INFO:registry:Loading language identifier...
INFO:registry:Loading transliteration model...
INFO:registry:Loading target model...
INFO:registry:Loading devanagari + sentiment model...
INFO:registry:All models ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Start frontend

```bash
cd frontend
npm run dev
```

### Open the app

```
http://localhost:5173
```

Paste a Nepali YouTube URL and click **Analyze**.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze comments from a YouTube URL |
| GET | `/api/health` | Check if all models are loaded |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/VIDEO_ID"}'
```

### Example response

```json
{
  "results": [
    {
      "original_comment": "ओलीले देश बिगारे",
      "language": "NE_DEV",
      "devanagari_comment": "ओलीले देश बिगारे",
      "targets": [
        {
          "target": "ओली",
          "aspect": "governance",
          "sentiment": "negative"
        }
      ],
      "skipped": false
    }
  ],
  "stats": {
    "total_comments": 10,
    "processed": 7,
    "skipped": 3,
    "sentiment_counts": {"negative": 4, "positive": 2, "neutral": 1},
    "aspect_counts": {"governance": 3, "policy": 2, "service": 2},
    "per_target": [
      {
        "target": "ओली",
        "mentions": 4,
        "breakdown": [
          {"aspect": "governance", "positive": 1, "negative": 3, "neutral": 0}
        ]
      }
    ]
  }
}
```

---

## Frontend Features

- URL input with Enter key support
- Loading spinner with estimated wait time
- Result cards showing original comment, Devanagari translation, detected language, targets, aspect and sentiment per target
- Skipped comments shown with reason (Code-Mixed, Too Short etc.)
- Overall sentiment pie chart
- Overall aspect bar chart
- Per-target breakdown charts for top 2 targets showing top 3 aspects with sentiment frequency

---

## Language Handling

| Language | Label | Handling |
|----------|-------|---------|
| Nepali Devanagari | NE_DEV | Processed directly |
| English | EN | Translated to Devanagari first |
| Roman Nepali | NE_ROM | Skipped |
| Code-Mixed | CODE_MIXED | Skipped |
| Too short (< 5 words) | TOO_SHORT | Skipped |

---

## Aspects

The model classifies comments into 5 aspects:

- **Governance** — leadership and administration
- **Policy** — laws and government decisions
- **Service** — public services and delivery
- **Corruption** — misconduct and bribery
- **Economy** — financial and economic matters

---

## Running on Google Colab (Alternative)

If local RAM is insufficient (requires ~8GB), run the backend on Google Colab:

1. Upload project files to Google Drive
2. Open [Google Colab](https://colab.research.google.com)
3. Mount Drive and run the FastAPI server
4. Use the Colab proxy URL in `App.jsx`
5. Run only the frontend locally with `npm run dev`

---

## Known Limitations

- Requires 8GB+ RAM to run all models locally
- Inference runs on CPU — each request takes 1-3 minutes
- Target model performance depends on comment clarity
- Roman Nepali comments are not currently supported

---



## Acknowledgements

- [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) for the transliteration model
- [Facebook FastText](https://fasttext.cc) for Nepali word embeddings
- [HuggingFace](https://huggingface.co) for XLM-RoBERTa

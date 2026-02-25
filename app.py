from fastapi import FastAPI, UploadFile, File
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from parser import parse
from clean_text import clean_text

app = FastAPI()

# Model and Tokenizer Setup
MODEL_NAME = "aarushi1112/bert-ner-model"

# Load tokenizer and model from your HF repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.eval()  # inference mode

# Get id2label mapping
id2label = model.config.id2label

# Prediction Function
def predict_ner(text: str):
    """
    Predicts NER tags for the given cleaned text.
    Returns list of tuples: (word, BIO-tag)
    """
    words = text.split()

    # Tokenize words with alignment
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids(batch_index=0)  

    results = []
    prev_word_idx = None

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue  
        results.append({
            "word": words[word_idx],
            "label": id2label[predictions[token_idx]]
        })
        prev_word_idx = word_idx

    return results

# /predict Route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded PDF temporarily
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse PDF to text using your parser.py
    raw_text = parse(file_path)

    # Clean the parsed text
    cleaned_text = clean_text(raw_text)

    # Get NER predictions
    ner_results = predict_ner(cleaned_text)

    return {"predictions": ner_results}

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


from flask import Flask, request, jsonify
import re
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Türkçe NER modeli ve tokenizer'ı yükleme
ner_model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner_tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="first")

# Türkçe sentiment analiz modeli ve tokenizer'ı yükleme
sentiment_model = AutoModelForSequenceClassification.from_pretrained("emre/turkish-sentiment-analysis")
sentiment_tokenizer = AutoTokenizer.from_pretrained("emre/turkish-sentiment-analysis")

# Sentiment analizi için fonksiyon
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt")
    outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment_map = {0: "olumsuz", 1: "nötr", 2: "olumlu"}
    return sentiment_map[predicted_class]

# Metni işleme fonksiyonu
def preprocess_text(text):
    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    # Küçük harfe dönüştürme
    text = text.lower()
    # Sayıları çıkarma
    text = re.sub(r'\d+', '', text)
    return text

def extract_sentences_with_entities(text, entities):
    sentences = text.split('.')
    entity_sentences = {entity: "" for entity in entities}
    for sentence in sentences:
        for entity in entities:
            if entity in sentence:
                entity_sentences[entity] = sentence.strip()
    return entity_sentences

def process_text(text):
    ner_results = ner(text)
    entities = list(set([result['word'] for result in ner_results]))

    entity_sentences = extract_sentences_with_entities(text, entities)
    results = []
    for entity in entities:
        sentiment = analyze_sentiment(entity_sentences[entity])
        results.append({
            "entity": entity, 
            "sentiment": sentiment,
            "yorum": entity_sentences[entity]
        })

    output = {
        "entity_list": entities,
        "results": results
    }
    return output

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    result = process_text(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

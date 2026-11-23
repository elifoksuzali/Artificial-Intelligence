from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config["DEBUG"] = True  # Hata ayıklama modunu etkinleştirin

# NER modeli ve tokenizer'ı yükleme
ner_model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner_tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="first")

# Türkçe sentiment analiz modeli ve tokenizer'ı yükleme (Transformer modelleri)
sentiment_model = AutoModelForSequenceClassification.from_pretrained("emre/turkish-sentiment-analysis")
sentiment_tokenizer = AutoTokenizer.from_pretrained("emre/turkish-sentiment-analysis")

sent2_model = AutoModelForSequenceClassification.from_pretrained("anilguven/distilbert_tr_turkish_hotel_reviews")
sent2_tokenizer = AutoTokenizer.from_pretrained("anilguven/distilbert_tr_turkish_hotel_reviews")

# TF-IDF ve CountVectorizer modellerini yükleme
tf_vector = joblib.load('tf_vector.pkl')
ct_vector = joblib.load('ct_vector.pkl')
nb_model_tfidf = joblib.load('nb_model_tfidf.pkl')
nb_model_ct = joblib.load('nb_model_ct.pkl')

# Sentiment analizi için fonksiyon
def analyze_sentiment(text):
    # Transformer modelleri
    inputs1 = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs1 = sentiment_model(**inputs1)
    
    inputs2 = sent2_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs2 = sent2_model(**inputs2)
    
    logits1 = outputs1.logits
    probabilities1 = torch.softmax(logits1, dim=1)
    predicted_class1 = torch.argmax(probabilities1, dim=1).item()
    prob1 = probabilities1[0][predicted_class1].item()

    logits2 = outputs2.logits
    probabilities2 = torch.softmax(logits2, dim=1)
    predicted_class2 = torch.argmax(probabilities2, dim=1).item()
    prob2 = probabilities2[0][predicted_class2].item()

    # TF-IDF ve CountVectorizer için giriş verilerini hazırla
    tf_vec_text = tf_vector.transform([text])
    ct_vec_text = ct_vector.transform([text])

    predicted_class4 = nb_model_tfidf.predict(tf_vec_text)[0]
    predicted_class5 = nb_model_ct.predict(ct_vec_text)[0]

    prob4 = nb_model_tfidf.predict_proba(tf_vec_text)[0][predicted_class4]
    prob5 = nb_model_ct.predict_proba(ct_vec_text)[0][predicted_class5]

    sentiment_map = {0: "olumlu", 1: "nötr", 2: "olumsuz"}

    return {
        "model_1": {
            "class": sentiment_map[predicted_class1],
            "probability": prob1
        },
        "model_2": {
            "class": sentiment_map[predicted_class2],
            "probability": prob2
        },
        "model_4": {
            "class": sentiment_map[predicted_class4],
            "probability": prob4  # Naive Bayes modelinde doğrudan olasılık hesaplandı
        },
        "model_5": {
            "class": sentiment_map[predicted_class5],
            "probability": prob5  # Naive Bayes modelinde doğrudan olasılık hesaplandı
        }
    }

@app.route('/predict', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get("text")
        
        # NER analizi
        entities = ner(text)

        # Entity'ler için sentiment analizi sonuçlarını saklamak için liste
        results = []
        entity_list = []
        
        # Her entity için sentiment analizi yap ve sonuçları sakla
        for entity in entities:
            entity_text = entity['word']
            sentiment_results = analyze_sentiment(entity_text)

            best_model = max(sentiment_results, key=lambda x: sentiment_results[x]["probability"] if sentiment_results[x]["probability"] != "N/A" else 0)
            best_sentiment = sentiment_results[best_model]["class"]

            results.append({
                "entity": entity_text,
                "sentiment": best_sentiment
            })
            entity_list.append(entity_text)
        
        response = {
            "entity_list": entity_list,
            "results": results
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=0000)

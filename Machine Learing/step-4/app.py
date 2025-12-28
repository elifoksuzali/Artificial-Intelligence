from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
import json
from transformers import AutoTokenizer, AutoModel
import torch
from py2neo import Graph, Node, Relationship
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import threading
import time
import random
import string
import datetime  # Yeni dosya adı oluşturmak için
from flask import session

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
progress = 0

# Model ve tokenizer'ı yükledik
model = pickle.load(open('finalized_model.sav', 'rb'))
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

def filter(text):
    final_text = text.replace("<br />", " ")
    final_text = final_text.replace("  ", " ")
    return final_text

def feature_extraction(text):
    inputs = tokenizer(filter(text), return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[0][0].cpu().numpy()

def predict_sentiment(text):
    features = feature_extraction(text)
    pred = model.predict([features])[0]
    if pred == 0:
        return 'Olumsuz'
    elif pred == 1:
        return 'Tarafsız'
    else:
        return 'Olumlu'

def generate_random_username():
    # Trendyol kullanıcı adlarını paylaşmadığı için rastgele bir kullanıcı adı oluşturduk 
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

def load_to_neo4j(df):
    graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))

    # Üç merkez düğüm: Olumlu, Olumsuz, Tarafsız
    positive_node = Node("Sentiment", sentiment="Olumlu")
    negative_node = Node("Sentiment", sentiment="Olumsuz")
    neutral_node = Node("Sentiment", sentiment="Tarafsız")
    graph.create(positive_node)
    graph.create(negative_node)
    graph.create(neutral_node)

    for index, row in df.iterrows():
        # Yorum Node'u
        review_node = Node("Review", text=row['Review'], sentiment=row['Sentiment'])
        graph.create(review_node)

        # Kullanıcı Node'u 
        username = generate_random_username()
        user_node = Node("User", username=username)
        graph.create(user_node)

        # Yorumun sentiment'ına göre ilgili merkez düğüme bağlantı kurma işlemini gerçekleştirdik.
        sentiment = row['Sentiment']
        if sentiment == 'Olumlu':
            graph.create(Relationship(user_node, "YAZDI", positive_node))
        elif sentiment == 'Olumsuz':
            graph.create(Relationship(user_node, "YAZDI", negative_node))
        elif sentiment == 'Tarafsız':
            graph.create(Relationship(user_node, "YAZDI", neutral_node))

def clear_neo4j_data(graph):
    #Bir önceki analizle karışmaması için gerekli önlemleri aldık.
    graph.run("MATCH (n:Review) DETACH DELETE n")
    graph.run("MATCH (n:User) DETACH DELETE n")
    graph.run("MATCH (n:Sentiment) DETACH DELETE n")

@app.route('/')
def index():
    return render_template('index.html')

def scrape_comments(url, csv_filename):
    global progress
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.binary_location = "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
    service = Service("C:\\Users\\SERHAT\\Downloads\\geckodriver-v0.34.0-win32\\geckodriver.exe")

    driver = webdriver.Firefox(service=service, options=options)
    driver.get(url)

    scroll_pause_time = 2
    max_scrolls = 50
    scrolls = 0
    progress = 10

    last_height = driver.execute_script("return document.body.scrollHeight")

    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height
        scrolls += 1
        progress = min(90, int((scrolls / max_scrolls) * 100))

    html_content = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html_content, "html.parser")
    comments = soup.find_all("p")

    yorumlar = []
    for comment in comments:
        if not comment.has_attr("class") and not comment.has_attr("div") and not comment.has_attr("span") and not comment.has_attr("id") \
                and comment.text not in ["10.000’lerce yeni ürünü ve sezon trendlerini büyük indirimlerle yakalamak için,",
                                         "Sepetinizde Ürün Bulunmamaktadır.",
                                         "Popüler Marka ve Mağazalar",
                                         "Popüler Sayfalar"]:
            yorumlar.append({
                'User': generate_random_username(),  # Rastgele kullanıcı adı oluştur
                'Review': comment.text
            })

    df = pd.DataFrame(yorumlar)
    df['Sentiment'] = df['Review'].apply(predict_sentiment)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    load_to_neo4j(df)  # Yorumları Neo4j'e yükle
    progress = 100

@app.route('/scrape', methods=['POST'])
def scrape():
    global progress
    progress = 0
    url = request.form['url']
    
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"yorumlar_{current_datetime}.csv"
    session['current_datetime'] = current_datetime
    
    threading.Thread(target=scrape_comments, args=(url, csv_filename)).start()
    return jsonify({'message': 'Yorumlar çekiliyor, lütfen bekleyin...'})

@app.route('/result')
def result():
    # Session'dan current_datetime'i al
    current_datetime = session.get('current_datetime', None)
    
    if current_datetime:
        csv_filename = f"yorumlar_{current_datetime}.csv"
        df = pd.read_csv(csv_filename)

        # Duygu dağılımı hesaı
        olumlu_sayisi = df[df['Sentiment'] == 'Olumlu'].shape[0]
        tarafsiz_sayisi = df[df['Sentiment'] == 'Tarafsız'].shape[0]
        olumsuz_sayisi = df[df['Sentiment'] == 'Olumsuz'].shape[0]

        toplam = olumlu_sayisi + tarafsiz_sayisi + olumsuz_sayisi
        olumlu_yuzde = (olumlu_sayisi / toplam) * 100
        tarafsiz_yuzde = (tarafsiz_sayisi / toplam) * 100
        olumsuz_yuzde = (olumsuz_sayisi / toplam) * 100

        return render_template('result.html', olumlu=olumlu_yuzde, tarafsiz=tarafsiz_yuzde, olumsuz=olumsuz_yuzde,
                               olumlu_sayisi=olumlu_sayisi, tarafsiz_sayisi=tarafsiz_sayisi,
                               olumsuz_sayisi=olumsuz_sayisi, csv_filename=csv_filename)
    else:
        return "CSV dosyası bulunamadı."

@app.route('/progress')
def get_progress():
    global progress
    return jsonify({'progress': progress})

@app.route('/download')
def download():
    df = pd.read_csv("yorumlar.csv")
    graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))
    
    # Neo4j veri tabanını temizleyip verileri neo4'ye yükledik.
    clear_neo4j_data(graph)
    
    load_to_neo4j(df)
    
    return send_file("yorumlar.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

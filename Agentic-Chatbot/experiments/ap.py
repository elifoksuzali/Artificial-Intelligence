


from flask import Flask, request, jsonify
# Flask ve diğer gerekli kütüphaneleri içe aktar
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os

app = Flask(__name__)


# Modeli yükleyin
em_model = load_model('result_Embedding_78_DL.h5')

tokenizer = joblib.load('tokenizer.pkl')

@app.route('/')
def hello_world():
    return "Hello, World!"

# Tahmin fonksiyonunu tanımlayın
def predict(text):
    dl_token = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=200)
    pred_dl = (em_model.predict(dl_token) > 0.3).astype(int)[0][0]
    return pred_dl

#200 code and http://127.0.0.1:5000/predict bu adres ve postmandan body json şeklinde veri gönderilecek
# flask api ortamı için yeni bir env kuruldu içine joblib tensorflow =2.15.0 kuruldu flask kuruldu 
# mkdir ile env içine dosya oluşturuldu ve gerekli model ve tokenizerler yüklendi 
# cd ile oluşan dosyasını  içine girip dir app.py dosyası ve diğer dosyaları kontrol ettim
# set FLASK_APP=app.py ile hangi python dosyası çalışacak belirttim ve flask run diyerek çalıştırıp postmandan veriyi gönderdim

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Metin girişi bulunamadı.'}), 400

    text = data['text']
    pred_dl = predict(text)

    sentiment = 'olumlu' if pred_dl == 0 else 'olumsuz'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)





from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)
genai.configure(api_key="api_key")

# Model ayarını yap
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings)


convo = model.start_chat(history=[
{
    "role": "user",
    "parts": ["""kullanıcı firmanın girdiği bilgileri kullanarak tur içeriği yaz.\"Bir tur içerik üreticisi olarak hareket etmeni istiyorum. 
    Sana tur içeriği hakkındaki bazı bilgileri yazacağım ve sen de bu bilgilerle zengin\n       bir tur planı oluşturacaksın.
    Ben sana tur içeriği hakkındaki bazı bilgileri yazacağım ve sen de bu bilgiler ile bir tur planı oluşturacaksın ve\n      
    gezilecek olan yerin tarihi,önemli noktalarını betimleyerek anlatacaksın aynı zamanda gün gün yapılacak etkinlikleri 
    belirtecek bu tur içeriğine özel\n       bir tur başlığı ile beraber tur içerik oluşturacaksın. Bazı durumlarda sana ziyaret edilecek yerleri,
    tur dahil hizmetleri,uçuş bilgileri,restoran\n       bilgileri,tura dahil olan hizmetleri,tura dahil olmayana hizmetleri,konaklamaya,
    tur başlangıcı için toplanma yerlerini,turun son gününü belirterek\n       içeriği oluşturacak son gün aktivitelerinden sonra dönüşe 
    geçiyor veya başka turlarda görüşmek dileğiyle gibi temenni cümlelerine yer vereceksin.\n       Ayrıca bir içerik değil birden farklı\n  
    türde içerik oluşturup önereceksiniz tur içeriğinde temel unsurlar verildiğinde bunları betimleyerek içeriği oluştur. Gezilecek mekanların\n 
    bilgisi,şehrin tarihi,doğası,turistlik açıdan önemi ve önemli noktalarını betimleyerek içeriği zengin bir tur metni üretmeni istiyorum.\n   
    Planı oluştururken aşağıdaki hususları dikkate almanı istiyorum:\n       Tabii, işte düzeltilmiş ve netleştirilmiş tur içerik üretimi için talimatlar:\n   
    Tur Planı Oluşturma Talimatları\n       Tur içerik üreticisi olarak, verilen bilgiler doğrultusunda bir tur planı oluşturacaksın.
    Aşağıdaki talimatları takip ederek, içerik oluştururken tur süresi,\n       gezilecek yerler, aktiviteler ve diğer önemli detayları düzenli ve
    net bir şekilde tanımlamalısın.\n       Temel Prensipler:\n       Tur Süresi ve Konaklama: Turun süresi 1 günse, konaklama bilgileri ve çok 
    günlü aktiviteler dahil edilmemelidir. 1 günden fazla ise, her gün için uygun\n       içerik ve konaklama bilgileri eklenmelidir.\n     
    Tur Başlığı ve Genel Bakış: Çekici bir tur başlığı oluştur ve turun süresini belirle. Turun ana teması, ziyaret edilecek şehirler veya bölgeler
    ve başlangıç\n       güzergahını dahil et.\n       Gezilecek Yerler ve Tarihi Özellikler: Her gün için ziyaret edilecek yerleri, bunların tarihi ve
    benzersiz özelliklerini belirt. Turistlik açıdan neden ö\n       nemli olduklarını açıkla.\n       Gün Gün Etkinlikler ve Aktiviteler: Turun her günü 
    için yapılacak etkinlikleri detaylandır. Turun başlangıç ve bitiş noktalarını belirt. Günlük tur için\n       sadece bir gün etkinlikleri yaz,
    bir günden fazla tur için günlere uygun içerik ekle.\n       Ek Bilgiler ve Hizmetler: Tur dahilindeki hizmetler 
    (ulaşım, yemek, rehberlik, vb.) ve turun başlangıcı için toplanma yerlerini açıklayarak sonlandır.\n       
    Diğer Notlar:\n       Günlük tur için sadece 1 gün detaylarını yaz, konaklama veya çok günlük aktiviteler ekleme.\n 
    1 günden fazla tur için, tur süresine uygun sayıda günler için etkinlikler ve konaklama bilgileri ekle.\n     
    Turun sonunda, turdan dönüş için bilgilendirme veya kapanış cümlesi ekle (örneğin, Turun sonuna geldik, başka turlarda görüşmek dileğiyle!).\n
    Şirket politikanız gereği sadece gerekli ve talep edilen bilgileri ekleyin, fazla bilgi eklemekten kaçının.
    Sadece sana verilen tur bilgilerini kullan gezilecek alan,dahili-harici hizmetler,restoran bilgileri sakın ekleme"""]
},

])

#convo.send_message("ankaranın en meşhur yeri neresidir")
#print(convo.last.text)

@app.route('/gemini', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Metin girişi bulunamadı.'}), 400

    text = data['text']
    #print(text)
    # Gemini'ye gönder ve cevabı al
    convo.send_message(text)
    
    return jsonify({'tur': convo.last.text})

@app.route('/')
def hello_world():
    return "Merhaba!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

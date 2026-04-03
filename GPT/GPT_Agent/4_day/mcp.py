import requests
import base64,json

username = 'user'
password = 'password'

def format_tour_data(data):
    formatted_data = ""
    if data.get('success') == 1 and 'data' in data:
        for key, value in sorted(data['data'].items(), key=lambda x: int(x[1].get('gun', '0'))):
            gun = value.get('gun', 'Bilinmiyor')
            baslik = value.get('baslik', 'Başlık bulunamadı')
            program = value.get('program', 'Program bilgisi yok')
            formatted_data += f"\n\n{gun}. Gün: {baslik}\n{program}"
    else:
        formatted_data += 'Veri alırken bir sorun oluştu.'
    return formatted_data.strip()

def fetch_tour_price(tour_id, username, password):
    url = 'https://www.test.com/index.php'
    credentials = f'{username}:{password}'
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Accept': '*/*',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        's': 's',
        'action': 'tour',
        'data[id]': tour_id
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # HTTP hatalarını yakalar
        #print(f"API Yanıtı: {response.text}")  # Yanıtı ekrana yazdır
        try:
            data = response.json()
            return format_tour_data(data)
        except ValueError as e:
            return 'Yanıt JSON formatında değil: ' + str(e) + ' - ' + response.text
    except requests.RequestException as e:
        return f'HTTP Hatası: {e} - {response.text}'
    
    

def preprocess_message_and_fetch_data(message_content, username, password):
    try:
        data = json.loads(message_content)
        if 'id' in data:
            tour_id = data['id']
            return fetch_tour_price(tour_id, username, password)
        else:
            return message_content
    except json.JSONDecodeError:
        return message_content
    
    

    
    
    
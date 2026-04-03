from prompt.prompt_db import PromptManager
from typing import Dict
from pydict.model import QuestionCategory, QuestionExtraction, AgentTask
from datetime import datetime

prompt_manager = PromptManager()

def question_analysis(chat_id: str, request_data: Dict):
    """Basit soru analizi fonksiyonu"""
    prompt = prompt_manager.load_prompts(request_data.get('parametre'))
    lastMessage = request_data.get('lastMessage')
    print(f"🏢 [CHAT-{chat_id}] Last Message: {lastMessage}")
    return prompt, lastMessage

def get_tour_list(turtipi: str, date: str) -> str:
    """Tur listesi getirme fonksiyonu"""
    tour_list_data = {
        "yurt içi": [
            {
                "id": "83",
                "isim": "Ege Akdeniz Turu Otobüs ile 5 Gece İstanbul çıkışlı, İzmit Hareketli",
                "turkodu": "2041",
                "gecesayisi": "6",
                "geceleme": "6 Gece 7 Gün",
                "konaklama": "5 Gece Yarım Pansiyon",
                "ulasim": "Otobüs ile İstanbul - Ege & Akdeniz - İstanbul",
                "ziyaretedilecekyerler": "Aspendos Tiyatrosu - Köprülü Kanyon - Rafting (Ekstra) - Düden Şelalesi - Kaleiçi - Yivli Minare - Üçağız Köyü - Kekova Tekne Turu - Batık Şehir - Simena - Kaş - Kalkan - Saklıkent Kanyonu",
                "vizesiz": "Bu tur vizesizdir",
                "kesinkalkis": "Bu tur kesin kalkışlı değildir",
                "ulasimtipi": "Otobüslü",
                "turtipi": "yurt içi",
                "turKategori": [{"isim": "Kültür Turları", "puan": "0"}]
            },
            {
                "id": "77",
                "isim": "Karadeniz Yaylalar Batum Turu Otobüs ile 6 Gece İstanbul çıkışlı, İzmit, Bolu hareketli",
                "turkodu": "2036",
                "gecesayisi": "6",
                "geceleme": "6 Gece 7 Gün",
                "konaklama": "3* Otellerde 5 Gece Yarım Pansiyon",
                "ulasim": "Otobüs ile İstanbul - Karadeniz - Batum - İstanbul",
                "ziyaretedilecekyerler": "Amasya - Ordu - Giresun - Trabzon - Maçka - Batum - Hopa - Borçka - Rize - Çamlıhemşin...",
                "vizesiz": "Bu tur vizesizdir",
                "kesinkalkis": "Bu tur kesin kalkışlı değildir",
                "ulasimtipi": "Otobüslü",
                "turtipi": "yurt içi",
                "turKategori": [{"isim": "Batum Turları", "puan": "0"}]
            }
        ]
    }
    
    if turtipi.lower() in tour_list_data:
        tours = tour_list_data[turtipi.lower()]
        if tours:
            result = f"📋 {turtipi.title()} Tur Listesi ({date}):\n\n"
            for i, tour in enumerate(tours, 1):
                result += f"🏖️ **Tur {i}:**\n"
                result += f"   • İsim: {tour['isim']}\n"
                result += f"   • Tur Kodu: {tour['turkodu']}\n"
                result += f"   • Süre: {tour['geceleme']}\n"
                result += f"   • Konaklama: {tour['konaklama']}\n"
                result += f"   • Ulaşım: {tour['ulasim']}\n"
                result += f"   • Ziyaret Edilecek Yerler: {tour['ziyaretedilecekyerler']}\n"
                result += f"   • Vize Durumu: {tour['vizesiz']}\n"
                result += f"   • Kalkış Durumu: {tour['kesinkalkis']}\n"
                result += f"   • Kategori: {', '.join([cat['isim'] for cat in tour['turKategori']])}\n\n"
            return result
        else:
            return f"❌ {turtipi} için {date} tarihinde tur bulunamadı."
    else:
        return f"❌ {turtipi} tur tipi için veri bulunamadı."

def analyze_message(message: str) -> Dict:
    """Mesaj analizi yapan basit fonksiyon"""
    message_lower = message.lower()
    
    # Kategori analizi
    if any(word in message_lower for word in ['merhaba', 'selam', 'hey']):
        category = "greeting"
        confidence = 0.9
    elif any(word in message_lower for word in ['tur', 'turlar', 'seyahat']):
        category = "tour_list"
        confidence = 0.8
    elif any(word in message_lower for word in ['fiyat', 'ücret', 'para']):
        category = "price"
        confidence = 0.7
    else:
        category = "unknown"
        confidence = 0.5
    
    # Basit çıkarma analizi
    extraction = {
        "turtipi": "yurt içi" if "yurt içi" in message_lower else None,
        "geceleme": None,
        "confidence": confidence,
        "reasoning": f"Mesaj '{category}' kategorisine ait olarak sınıflandırıldı"
    }
    
    return {
        "category": category,
        "confidence": confidence,
        "extraction": extraction
    }
from typing import Dict
from prompt.prompt_db import PromptManager
from datetime import datetime
from analysis.question_analysis import analyze_message

prompt_manager = PromptManager()

def process_chat_message(chat_id: str, request_data: Dict) -> Dict:
    """
    Chat mesajını işleyen ana fonksiyon
    """
    try:
        # Ana parametreleri al
        parametre = request_data.get('parametre')        
        last_message = request_data.get('lastMessage', '')
        
        # Prompt'ları yükle
        prompts = prompt_manager.load_prompts(parametre)
        
        # Mesaj analizi yap
        analysis_result = analyze_message(last_message)
        
        # Agent yanıtını oluştur
        agent_response = generate_agent_response(analysis_result, last_message)
        
        return {
            "status": "success",
            "chat_id": chat_id,
            "prompt": prompts,
            "parametre": parametre,
            "agent_response": agent_response,
            "agent_type": analysis_result["category"],
            "category_analysis": {
                "category": analysis_result["category"],
                "confidence": analysis_result["confidence"]
            },
            "extraction_analysis": analysis_result["extraction"],
            "confidence": analysis_result["confidence"],
            "original_message": last_message,
            "processed_at": datetime.now().isoformat()
        }
            
    except Exception as e:
        print(f"❌ [CHAT-{chat_id}] HATA: {str(e)}")
        return {
            "status": "error",
            "chat_id": chat_id,
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }

def generate_agent_response(analysis_result: Dict, original_message: str) -> str:
    """Analiz sonucuna göre agent yanıtı oluşturur"""
    category = analysis_result["category"]
    
    if category == "greeting":
        return "Merhaba! Size nasıl yardımcı olabilirim? Tur bilgileri, fiyatlar veya rezervasyon konularında sorularınızı yanıtlayabilirim."
    
    elif category == "tour_list":
        return "Tur listesi için size yardımcı olabilirim. Hangi tür tur arıyorsunuz? (Yurt içi, yurt dışı, kaç günlük vb.)"
    
    elif category == "price":
        return "Fiyat bilgileri için hangi tur hakkında bilgi almak istiyorsunuz? Tur kodu veya tur adını belirtirseniz size yardımcı olabilirim."
    
    else:
        return "Anlamadım, lütfen tekrar sorar mısınız? Tur bilgileri, fiyatlar veya rezervasyon konularında size yardımcı olabilirim."

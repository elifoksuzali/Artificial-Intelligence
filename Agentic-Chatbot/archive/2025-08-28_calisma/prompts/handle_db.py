# prompts/handle_db.py

import mysql.connector as mysql
from mysql.connector import Error
import logging
from db_key.db import DB_CONFIG

PROMPT_NAMES = [
    "repl_agent_system_prompt",
    "query_router_agent_system_prompt",
    "question_analyzer_system_prompt",
    "response_validator_agent_system_prompt",
    "determine_agent_and_extract_system_prompt",
    "system_prompt",
    "follow_up_agent_system_prompt",
    "get_agent_response",
    "get_tour_details_prompt",
    "qdrant_search_system_prompt",
    "agent_map_system_prompt",
    "chat_history_system_prompt",
    "mcp_service_agent_system_prompt",
    "greeting_agent_system_prompt",
    "query_correction_agent_system_prompt",
    "tour_analysis_agent_system_prompt"
]

# Manuel prompt tanımları (veritabanından yüklenemezse kullanılacak)
MANUAL_PROMPTS = {
    "query_correction_agent_system_prompt": """Sen bir yazım hatası düzeltme uzmanısın. Kullanıcının mesajındaki yazım hatalarını düzelt ve anlamı koru. Sadece yazım hatalarını düzelt, anlamı değiştirme. Yanıt olarak düzeltilmiş metni ver.""",
    
    "question_analyzer_system_prompt": """Sen bir soru analiz uzmanısın. Gelen soruyu analiz ederek tipini belirle. Yanıtını aşağıdaki JSON formatında ver:
    {"type": "greeting" veya "tour_info" veya "tour_list" veya "other", "detail_field": "alan_adı", "category": "kategori"}
    
    - greeting: Selamlaşma, teşekkür gibi genel mesajlar
    - tour_info: Belirli bir tur hakkında bilgi isteği
    - tour_list: Tur listesi isteği
    - tour_more: Daha fazla tur örneğine ait liste isteği
    - other: Diğer türdeki sorular""",
    
    "tour_analysis_agent_system_prompt": """Sen bir tur analiz uzmanısın. Kullanıcının sorusundan tur alanlarını çıkar. Yanıtını aşağıdaki JSON formatında ver:
    {"gecesayisi": "3", "turtipi": "yurt içi", "ulasimtipi": "otobüs", "vizesiz": "evet", "kesinkalkis": "İstanbul", "turkodu": "TR001", "konaklama": "3* otel", "ziyaretedilecekyerler": "İstanbul, Bursa"}
    
    ÖNEMLİ ALANLAR:
    - gecesayisi: Gece sayısı (örn: "3 günlük" -> "3")
    - turtipi: Tur tipi (yurt içi, yurt dışı)
    - ulasimtipi: Ulaşım tipi (otobüs, uçak, tren)
    - vizesiz: Vize durumu (evet, hayır)
    - kesinkalkis: Kalkış şehri
    - turkodu: Tur kodu
    - konaklama: Konaklama tipi (örn: "3* otel", "4* otel", "5* otel", "pansiyon", "yarım pansiyon")
    - ziyaretedilecekyerler: Ziyaret edilecek yerler
    
    ÖZEL DURUMLAR:
    - "3* otel" -> konaklama: "3* otel"
    - "5 yıldızlı otel" -> konaklama: "5* otel"
    - "yarım pansiyon" -> konaklama: "yarım pansiyon"
    - "tam pansiyon" -> konaklama: "tam pansiyon"
    
    Sadece sorusunda geçen alanları doldur, olmayanları ekleme. Alan adları küçük harfle yazılmalı. 
    ÖNEMLİ: Yanıtınızı geçerli JSON formatında verin, single quote kullanmayın, double quote kullanın.
    
    ÖRNEKLER:
    - "3 günlük yurt içi turları" -> {"gecesayisi": "3", "turtipi": "yurt içi"}
    - "3* otel konaklamalı turlar" -> {"konaklama": "3* otel"}
    - "5 yıldızlı otelde 7 gün konaklama" -> {"gecesayisi": "7", "konaklama": "5* otel"}
    - "yarım pansiyon yurt dışı turları" -> {"turtipi": "yurt dışı", "konaklama": "yarım pansiyon"}""",
    
    "repl_agent_system_prompt": """Sen bir tur danışmanısın. Kullanıcının sorularına detaylı ve yardımcı yanıtlar ver. Tur bilgilerini açık ve anlaşılır şekilde sun.

    🌟 Temel Kurallar:
    - Her zaman kullanıcının adıyla (firstName) ve cinsiyetine uygun şekilde hitap et
    - Önceki mesajları (previousMessages) dikkate al ve bağlamı koru
    - Son mesajı (lastMessage) yanıtla
    - Firma bilgilerini (companyId) kullan
    - Chat geçmişini (chatId) takip et
    
    🌟 Yanıt Formatı:
    - Selamlama: "Merhaba [Ad] [Bey/Hanım], ben MNG Asistan HarbiAI. Size nasıl yardımcı olabilirim?"
    - Tur listesi: "🌟 Sizin için özel seçimlerimiz:\n1. [Tur Adı]\n2. [Tur Adı]\n\n✨ Bu turlardan hangisi hakkında detaylı bilgi almak istersiniz?"
    - Genel yanıtlar: Kısa, net ve yardımcı ol
    
    🌟 Önemli Notlar:
    - Asla içeriği değiştirme ve açıklama yapma
    - Sadece yanıt ver
    - Türkçe karakterleri doğru kullan
    - Profesyonel ve samimi bir dil kullan"""
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PromptManager:
    def __init__(self):
        self.prompts = {}
        self.parametre = None

    async def load_prompts(self, parametre: str) -> dict:
        print(f"[PromptManager] Parametre alındı: {parametre}")
        aitip = "gpt-4.1" if parametre == "pro" else "gpt-4o-mini" if parametre == "free" else None
        if aitip is None:
            logging.warning(f"[PromptManager] Geçersiz parametre: {parametre}")
            return {}

        self.parametre = parametre

        try:
            conn = mysql.connect(**DB_CONFIG, charset="utf8")
            if not conn.is_connected():
                logging.error("[PromptManager] Veritabanına bağlanılamadı.")
                # Veritabanı bağlantısı yoksa manuel prompt'ları kullan
                self.prompts = MANUAL_PROMPTS.copy()
                logging.info(f"[PromptManager] Manuel prompt'lar yüklendi: {len(self.prompts)}")
                return self.prompts

            cursor = conn.cursor()
            cursor.execute("""
                SELECT tip, prompt FROM kurum_ai_prompt
                WHERE aciklama = 'harbiAI' AND aitip = %s
            """, (aitip,))
            results = cursor.fetchone()

            self.prompts = {
                row[0]: row[1]
                for row in results
                if row[0] in PROMPT_NAMES
            }

            # Eksik prompt'ları manuel olarak ekle
            for prompt_name in PROMPT_NAMES:
                if prompt_name not in self.prompts and prompt_name in MANUAL_PROMPTS:
                    self.prompts[prompt_name] = MANUAL_PROMPTS[prompt_name]
                    logging.info(f"[PromptManager] Manuel prompt eklendi: {prompt_name}")

            logging.info(f"[PromptManager] {len(self.prompts)} prompt yüklendi.")
            return self.prompts

        except Error as e:
            logging.error(f"[PromptManager] SQL Hatası: {e}")
            # Hata durumunda manuel prompt'ları kullan
            self.prompts = MANUAL_PROMPTS.copy()
            logging.info(f"[PromptManager] Manuel prompt'lar yüklendi: {len(self.prompts)}")
            return self.prompts

        finally:
            if conn:
                conn.close()

    def get(self, prompt_name: str) -> str:
        return self.prompts.get(prompt_name, "")

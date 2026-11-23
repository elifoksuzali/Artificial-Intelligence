import mysql.connector as mysql
from mysql.connector import Error
import logging
import openai


from config import DB_USER,DB_PASSWORD,DB_NAME,DB_HOST

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}

class KeyManager:
    
    def __init__(self):
        pass

    def load_api_key(self) -> dict:
        try:
            conn = mysql.connect(**DB_CONFIG, charset="utf8")
            if not conn.is_connected():
                logging.error("[PromptManager] Veritabanına bağlanılamadı.")
                return {}

            cursor = conn.cursor()
            cursor.execute("SELECT anahtar, deger FROM ayarlar WHERE anahtar IN ('gpt_api_key', 'gemini_api_key','qdrant_api_key','qdrant_host')")
            results = cursor.fetchall()
          
            cursor.close()
            conn.close()
            
            # Sonuçları dictionary'ye çevir
            api_keys = {}
            for row in results:
                api_keys[row[0]] = row[1]
            
            return api_keys

        except Exception as e:
            logging.error(f"[PromptManager] Veritabanı hatası: {str(e)}")
            return {}
        
if __name__ == "__main__":
    key_manager = KeyManager()
    print(key_manager.load_api_key())
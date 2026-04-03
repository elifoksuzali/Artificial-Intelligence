import mysql.connector as mysql
from mysql.connector import Error
import logging
import openai

DB_CONFIG = {
    "host": "host",
    "user": "user",
    "password": "password",
    "database": "database"
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
            cursor.execute("SELECT anahtar, deger FROM ayarlar WHERE anahtar IN ('gpt_api_key', 'gemini_api_key')")
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


import mysql.connector as mysql
from mysql.connector import Error
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setting import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME


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
                logging.error("[KeyManager] Veritabanına bağlanılamadı.")
                return {}

            cursor = conn.cursor()
            cursor.execute("""
                        SELECT anahtar, deger
                        FROM ayarlar 
                        WHERE anahtar IN (
                        'SocialChatbotWhiteList',
                        'gpt_api_key', 
                        'gemini_api_key',
                        'qdrant_api_key',
                        'qdrant_host',
                        'ollama_api_key',
                        'system_room_base_url',
                        's13_qdrant_api_key',
                        's13_qdrant_db.example.com',
                        'model_gpt',
                        'model_ollama',
                        'haskey_s13',
                        'mng_api_url'
                        )""")
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Sonuçları dictionary'ye çevir
            api_keys = {}
            for row in results:
                api_keys[row[0]] = row[1]
            
            return api_keys

        except Exception as e:
            logging.error(f"[KeyManager] Veritabanı hatası: {str(e)}")

            
            return {}
        
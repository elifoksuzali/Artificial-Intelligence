import mysql.connector as mysql
from mysql.connector import Error
import logging

DB_CONFIG = {
    "host": "host",
    "user": "user",
    "password": "password",
    "database": "database"
}
class PromptManager:
    def __init__(self):
        self.parametre = None

    def load_prompts(self, parametre: str) -> dict:
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
                return {}

            cursor = conn.cursor()
            cursor.execute("""
                SELECT tip, prompt FROM kurum_ai_prompt
                WHERE aciklama = 'harbiAI' AND aitip = %s
            """, (aitip,))
            results = cursor.fetchall()
            prompts = {tip: prompt for tip, prompt in results}
            print(f"[PromptManager] {len(prompts)} prompt yüklendi.")
            cursor.close()
            conn.close()
            return prompts

        except Exception as e:
            logging.error(f"[PromptManager] Veritabanı hatası: {str(e)}")
            return {}
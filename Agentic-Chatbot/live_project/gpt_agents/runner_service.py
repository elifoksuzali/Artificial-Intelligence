"""
OpenAI API üzerinden çalışan Agent runner.
agent_parametre="pro" olduğunda bu modül kullanılır.
"""
import json
import logging
import sys
import os

# Parent dizini import path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import AsyncOpenAI  # pip paketi (gpt_agents değil)
from token_price import Usage
from connection_sql import KeyManager

logger = logging.getLogger(__name__)

# =========================================================
# API KEY & MODEL (veritabanından)
# =========================================================

key_manager = KeyManager()
api_list = key_manager.load_api_key()

OPENAI_API_KEY = api_list.get('gpt_api_key', '')
OPENAI_MODEL = api_list.get('model_gpt','')

if not OPENAI_API_KEY:
    logger.warning(
        "⚠️ OpenAI API key bulunamadı! "
        "Veritabanında 'gpt_api_key' ayarını kontrol edin."
    )
else:
    logger.info(f"✅ OpenAI API key yüklendi (model: {OPENAI_MODEL})")

# OpenAI async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# AGENT CLASS
# =========================================================

class Agent:
    """
    OpenAI API ile çalışan Agent sınıfı.
    Ollama runner_service.py ile aynı interface'e sahip.
    """
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = None,
    ):
        """
        Args:
            name: Agent adı (loglama için)
            instructions: System prompt
            model: OpenAI model adı (None ise global OPENAI_MODEL kullanılır)
        """
        self.name = name
        self.instructions = instructions
        self.model = model or OPENAI_MODEL

    async def run(
        self, 
        user_input: str, 
        chat_history: list[dict] = None
    ) -> dict:
        """
        OpenAI API'ye istek gönderir ve yanıt döner.
        
        Args:
            user_input: Kullanıcı mesajı
            chat_history: Önceki mesajlar (OpenAI format)
        
        Returns:
            dict: {"text": str, "usage": Usage} veya parsed JSON
        """
        chat_history = chat_history or []

        # Mesajları hazırla
        messages = [{"role": "system", "content": self.instructions}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})

        try:
            # OpenAI API çağrısı
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            text = response.choices[0].message.content or ""
            
            # Token kullanımı
            usage = Usage(
                requests=1,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            logger.info(
                f"[Agent: {self.name}] OpenAI ({self.model}) yanıt aldı "
                f"({usage.total_tokens} token)"
            )

            # JSON parse deneme
            # TOON formatı [#ID] JSON olarak parse edilmemeli
            stripped = text.strip()
            looks_like_json = (
                stripped.startswith("{") 
                or (stripped.startswith("[") and not stripped.startswith("[#"))
            )
            
            if looks_like_json:
                try:
                    parsed = json.loads(text)
                    parsed["usage"] = usage
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(
                        f"[Agent: {self.name}] JSON parse başarısız: {e}"
                    )

            return {"text": text, "usage": usage}

        except Exception as e:
            logger.error(
                f"[Agent: {self.name}] OpenAI hatası: {type(e).__name__}: {e}"
            )
            raise

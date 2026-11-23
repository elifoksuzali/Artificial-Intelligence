import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from db_key.db import KeyManager

class SimpleAgent:
    def __init__(self, name: str, model: str, instructions: str):
        self.name = name
        self.model = model
        self.instructions = instructions
        
        # API key'i veritabanından al
        key_manager = KeyManager()
        api_keys = key_manager.load_api_key()
        openai_api_key = api_keys.get('gpt_api_key', '')
        
        if not openai_api_key:
            print(f"⚠️ {self.name} için OpenAI API key bulunamadı!")
            self.client = None
        else:
            self.client = OpenAI(api_key=openai_api_key)
        
    async def process_message(self, message: str, context: List[Dict[str, str]] = None) -> str:
        """
        Mesajı işler ve yanıt döner
        """
        try:
            # API key kontrolü
            if not self.client:
                return "Üzgünüm, OpenAI API key yapılandırılmamış."
            
            # Context'i hazırla
            messages = []
            
            # System message ekle
            messages.append({
                "role": "system",
                "content": self.instructions
            })
            
            # Context varsa ekle
            if context:
                messages.extend(context)
            
            # Kullanıcı mesajını ekle
            messages.append({
                "role": "user",
                "content": message
            })
            
            # OpenAI API çağrısı
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ {self.name} hatası: {str(e)}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"
    
    async def run(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None, input_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Eski kodla uyumlu run metodu
        """
        try:
            # API key kontrolü
            if not self.client:
                return {"answer": "OpenAI API key yapılandırılmamış", "source": "error", "confidence": 0.0}
            
            # Mesajları hazırla
            system_messages = []
            
            # Agent talimatlarını ekle - JSON formatını zorlamayalım
            system_messages.append({
                "role": "system",
                "content": f"{self.instructions}\n\nÖNEMLİ: Yanıtınızı doğal dilde verin. JSON formatını açık şekilde göstermeyin. Eğer anlamsız bir mesaj alırsanız, nazikçe daha detay vermesini isteyin."
            })
            
            # Context varsa ekle
            if context:
                if isinstance(context, dict):
                    for key, value in context.items():
                        if isinstance(value, set):
                            context[key] = list(value)
                context_message = f"Context: {json.dumps(context, ensure_ascii=False)}"
                system_messages.append({"role": "system", "content": context_message})
            
            # input_data varsa ekle
            if input_data:
                input_data_message = f"input_data: {json.dumps(input_data, ensure_ascii=False)}"
                system_messages.append({"role": "system", "content": input_data_message})
            
            # Tüm mesajları birleştir
            all_messages = system_messages + messages
            
            # OpenAI API çağrısı - JSON formatını zorlamayalım
            response = self.client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Yanıtı al
            content = response.choices[0].message.content
            if not content:
                return {"answer": "Cevap alınamadı", "source": "gpt", "confidence": 0.0}
            
            # Yanıtı temizle ve kontrol et
            content = content.strip()
            
            # Eğer yanıt JSON formatında başlıyorsa, sadece answer kısmını al
            if content.startswith('{') and content.endswith('}'):
                try:
                    parsed = json.loads(content)
                    if "answer" in parsed:
                        content = parsed["answer"]
                    elif "response" in parsed:
                        content = parsed["response"]
                    elif "message" in parsed:
                        content = parsed["message"]
                    else:
                        # JSON'da answer yoksa, tüm içeriği string olarak kullan
                        content = str(parsed)
                except json.JSONDecodeError:
                    # JSON parse edilemezse, içeriği olduğu gibi kullan
                    pass
            
            # Eğer yanıt çok kısaysa veya anlamsızsa, daha detay iste
            if len(content) < 10 or content.lower() in ['test', 'ok', 'evet', 'hayır', 'bilmiyorum']:
                content = "Mesajınızı tam olarak anlayamadım. Size daha iyi yardımcı olabilmem için biraz daha detay verebilir misiniz?"
            
            # Sonuç döndür
            return {
                "answer": content,
                "source": "gpt", 
                "confidence": 0.8
            }
            
        except Exception as e:
            print(f"❌ {self.name} run hatası: {str(e)}")
            return {"answer": f"Bir hata oluştu: {str(e)}", "source": "error", "confidence": 0.0}
    
    def __str__(self):
        return f"SimpleAgent({self.name}, {self.model})"
    
    def __repr__(self):
        return self.__str__()


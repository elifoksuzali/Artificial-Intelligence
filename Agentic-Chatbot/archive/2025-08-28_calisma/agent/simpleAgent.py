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
            
            # Question Analyzer Agent için özel talimat
            if "Question Analyzer" in self.name:
                system_messages.append({
                    "role": "system",
                    "content": f"{self.instructions}\n\nÖNEMLİ: Yanıtınızı aşağıdaki JSON formatında verin:\n{{\"type\": \"greeting\" veya \"tour_info\" veya \"tour_list\" veya \"other\", \"detail_field\": \"alan_adı\", \"category\": \"kategori\"}}"
                })
            else:
                # Diğer agent'lar için normal talimat
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
            
            # Reply Agent için özel işlem - extracted_items varsa onları kullan
            if "Reply Agent" in self.name and context and "extracted_items" in context:
                # Sadece çıkarılan öğeleri kullan, tüm input_data'yı değil
                extracted_items = context["extracted_items"]
                selected_agent = context.get("selected_agent", "unknown")
                
                # Çıkarılan öğelere göre filtrelenmiş veri oluştur
                if input_data and extracted_items:
                    filtered_data = []
                    for item in input_data:
                        # extracted_items'da belirtilen indekslere göre filtrele
                        if isinstance(extracted_items, list):
                            for idx in extracted_items:
                                if isinstance(idx, int) and 0 <= idx < len(input_data):
                                    filtered_data.append(input_data[idx])
                                elif isinstance(idx, str):
                                    # String olarak gelen öğeler için (örn: tur adı)
                                    if idx in str(item):
                                        filtered_data.append(item)
                    
                    if filtered_data:
                        input_data_message = f"Filtrelenmiş veri (Agent: {selected_agent}): {json.dumps(filtered_data, ensure_ascii=False)}"
                        system_messages.append({"role": "system", "content": input_data_message})
                    else:
                        # Filtreleme başarısız olursa orijinal input_data'yı kullan
                        input_data_message = f"input_data: {json.dumps(input_data, ensure_ascii=False)}"
                        system_messages.append({"role": "system", "content": input_data_message})
                else:
                    # input_data yoksa sadece context'i kullan
                    pass
            else:
                # Diğer agent'lar için normal input_data işlemi
                if input_data:
                    input_data_message = f"input_data: {json.dumps(input_data, ensure_ascii=False)}"
                    system_messages.append({"role": "system", "content": input_data_message})
            
            # Tüm mesajları birleştir
            all_messages = system_messages + messages
            
            # OpenAI API çağrısı
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
            
            # Question Analyzer Agent için JSON parse et
            if "Question Analyzer" in self.name:
                try:
                    # JSON formatında yanıt döndür
                    if content.startswith('{') and content.endswith('}'):
                        parsed = json.loads(content)
                        return {
                            "answer": content,  # JSON string olarak sakla
                            "source": "gpt",
                            "confidence": 0.8
                        }
                    else:
                        # JSON değilse, basit bir JSON oluştur
                        return {
                            "answer": '{"type": "unknown", "detail_field": null, "category": null}',
                            "source": "gpt",
                            "confidence": 0.8
                        }
                except json.JSONDecodeError:
                    # JSON parse edilemezse, basit bir JSON oluştur
                    return {
                        "answer": '{"type": "unknown", "detail_field": null, "category": null}',
                        "source": "gpt",
                        "confidence": 0.8
                    }
            
            # Diğer agent'lar için eski mantık
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
                # Query Correction Agent için bu kontrolü yapma
                if "Query Correction" not in self.name:
                    content = "Mesajınızı tam olarak anlayamadım. Size daha iyi yardımcı olabilmem için biraz daha detay verebil misiniz?"
            
            # Query Correction Agent için özel kontrol
            if "Query Correction" in self.name:
                print(f"🔍 Query Correction Agent çalışıyor...")
                print(f"   Orijinal mesaj: {messages[-1]['content']}")
                print(f"   Agent yanıtı: {content}")
                
                # Eğer yanıt orijinal mesajdan çok farklıysa, sadece yazım hatalarını düzelt
                original = messages[-1]['content'].lower()
                corrected = content.lower()
                
                # Basit yazım hatası kontrolü
                if len(corrected) > len(original) * 2 or len(corrected) < len(original) * 0.5:
                    print(f"   ⚠️ Yanıt çok farklı, sadece yazım hatası düzeltiliyor")
                    # Orijinal mesajı döndür (yazım hatası yoksa)
                    return {
                        "answer": messages[-1]['content'],
                        "source": "query_correction",
                        "confidence": 0.9
                    }
                
                # Eğer yanıt anlamı değiştiriyorsa, orijinali döndür
                if "yardımcı olabilmem" in corrected or "detaylı paylaşır mısınız" in corrected:
                    print(f"   ⚠️ Yanıt anlamı değiştiriyor, orijinal döndürülüyor")
                    return {
                        "answer": messages[-1]['content'],
                        "source": "query_correction",
                        "confidence": 0.9
                    }
                
                print(f"   ✅ Yazım hatası düzeltildi")
                return {
                    "answer": content,
                    "source": "query_correction",
                    "confidence": 0.9
                }
            
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


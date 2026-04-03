from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import mysql.connector
from mysql.connector import Error
import sys
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import openai
from qdrant_client import QdrantClient
from mcp import fetch_tour_price, username, password
from database.db import KeyManager
# main.py'dan gerekli importları yap
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    ConversationContext,
    get_context_for_question,
    query_router_agent,
    reply_agent,
    input_data,
    question_analyzer_agent,
    SocialRequest,
    SocialResponse,
    ResponseData,
    response_validator_agent
)

km = KeyManager()
api_key = km.load_api_key()

app = FastAPI(title="Social Chatbot API")
openai.api_key = api_key['gpt_api_key']

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veritabanı bağlantı bilgileri
DB_CONFIG = {
    'host': 'host',
    'user': 'user',
    'password': 'password',
    'database': 'database'
}

# Qdrant bağlantısı
qdrant_client = None  # Global değişken olarak tanımla

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient()
    return qdrant_client

def get_SocialChatbotWhiteList():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("SELECT deger FROM ayarlar WHERE anahtar = 'anahtar'")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                # Virgülle ayrılmış IP'leri listeye çevir
                whitelist_ips = [ip.strip() for ip in result[0].split(',')]
                return whitelist_ips
            return []
    except Error as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        return []  # veya uygun bir hata mesajı döndürün
            
# IP kontrolü için middleware
@app.middleware("http")
async def verify_ip(request: Request, call_next):
    # X-Forwarded-For header'ını kontrol et
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
    
    print(f"Gelen istek IP'si: {client_ip}")
    print(f"X-Forwarded-For: {forwarded_for}")
    print(f"Client IP: {request.client.host}")
    
    whitelist_ips = get_SocialChatbotWhiteList()
    print(f"Whitelist IP'ler: {whitelist_ips}")
    
    if client_ip not in whitelist_ips:
        error_msg = f"IP is not in whitelist: {client_ip}"
        print(f"Hata: {error_msg}")
        return JSONResponse(
            status_code=403,
            content={"success": False, "errorMessage": error_msg}
        )
    
    response = await call_next(request)
    return response
        
# Her istek için yeni bir ConversationContext oluştur
conversation_contexts: Dict[str, ConversationContext] = {}

@app.post("/social-chatbot/message", response_model=SocialResponse)
async def process_message(request: SocialRequest):
    global input_data
    
    try:
        print("\n=== YENİ MESAJ İŞLEME BAŞLADI ===")
        print(f"Gelen mesaj: {request.lastMessage}")
        
        # Session ID olarak chatId kullan
        session_id = request.chatId
        print(f"Session ID (chatId): {session_id}")
        
        # Session için context oluştur veya mevcut olanı kullan
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = ConversationContext()
            print("Yeni conversation context oluşturuldu")
        else:
            print("Mevcut conversation context kullanılıyor")
        
        conv_context = conversation_contexts[session_id]
        
        # Chat history'yi oluştur
        chat_history = []
        
        # Müşteri bilgilerini ekle
        customer_info = {
            "role": "system",
            "content": f"Kullanıcı Bilgileri:\nAd: {request.customer.firstName}\nSoyad: {request.customer.lastName}\nCinsiyet: {request.customer.gender}"
        }
        chat_history.append(customer_info)
        
        # Önceki mesajları ekle ve son tur listesini bul
        last_tour_list = None
        for msg in request.previousMessages:
            chat_history.append({
                "role": "user" if msg.type == "USER" else "assistant",
                "content": msg.content
            })
            # Son asistan mesajında tur listesi var mı kontrol et
            if msg.type == "ASSISTANT" and "1." in msg.content and "2." in msg.content:
                last_tour_list = msg.content
        
        # Son mesajı ekle
        chat_history.append({
            "role": "user",
            "content": request.lastMessage
        })
        
        # Son tur listesini parse et ve last_shown_tours'u güncelle
        if last_tour_list:
            print("\n📝 Son tur listesi bulundu, parse ediliyor...")
            tour_list = []
            lines = last_tour_list.split('\n')
            for line in lines:
                if line.strip() and line[0].isdigit():
                    # Tur numarasını ve ismini ayır
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        tour_number = int(parts[0].strip())
                        tour_name = parts[1].strip()
                        tour_list.append({
                            "id": f"TOUR_{tour_number}",  # Geçici ID
                            "payload": {
                                "isim": tour_name
                            }
                        })
            
            if tour_list:
                print(f"✅ {len(tour_list)} tur parse edildi")
                conv_context.last_shown_tours = tour_list
                print("Son gösterilen turlar güncellendi:")
                for i, tour in enumerate(tour_list, 1):
                    print(f"{i}. {tour['payload']['isim']}")
        
        # Önce soruyu analiz et
        analysis = await question_analyzer_agent.run(chat_history)

        # Analiz sonucunu JSON olarak parse et
        try:
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            elif isinstance(analysis, list):
                # Eğer liste dönerse, ilk elemanı al
                if analysis:
                    analysis = analysis[0] if isinstance(analysis[0], dict) else {"answer": {"type": "unknown"}}
                else:
                    analysis = {"answer": {"type": "unknown"}}
            
            analysis_type = analysis.get("answer", {}).get("type", "unknown")
        except json.JSONDecodeError:
            analysis_type = "unknown"
        except Exception as e:
            print(f"❌ Analiz hatası: {str(e)}")
            analysis_type = "unknown"
        
        # final_result'ı başlangıçta tanımla
        final_result = None
        
        # Eğer selamlama ise direkt reply_agent'ı kullan
        if analysis_type == "greeting":
            print("\n🤖 Selamlama yanıtı hazırlanıyor...")
            try:
                greeting_response = await reply_agent.run(chat_history)
                print(f"Selamlama yanıtı: {greeting_response}")
                print(f"Selamlama yanıt tipi: {type(greeting_response)}")
                
                # Eğer greeting_response liste ise, ilk elemanı al
                if isinstance(greeting_response, list):
                    print("⚠️ Selamlama agent'ı liste döndürdü, ilk eleman alınıyor...")
                    if greeting_response:
                        if isinstance(greeting_response[0], dict):
                            greeting_response = greeting_response[0]
                        else:
                            greeting_response = {"answer": str(greeting_response[0]), "source": "list_first", "confidence": 0.8}
                    else:
                        greeting_response = {"answer": "Boş selamlama yanıtı", "source": "empty_list", "confidence": 0.0}
                elif not isinstance(greeting_response, dict):
                    # Eğer dictionary değilse, string'e çevir
                    greeting_response = {"answer": str(greeting_response), "source": "converted", "confidence": 0.8}
                
                # Yanıtı kontrol et ve düzelt
                validation_result = await response_validator_agent.run([
                    {"role": "system", "content": "Yanıt kontrolü yapılacak"},
                    {"role": "user", "content": str(greeting_response)}
                ])
                
                # Eğer yanıt geçerli değilse, düzeltilmiş yanıtı kullan
                if isinstance(validation_result, dict):
                    if not validation_result.get("is_valid", True):
                        answer = validation_result.get("corrected_response", "Üzgünüm, mesajınızı tam olarak anlayamadım. Size daha iyi yardımcı olabilmem için biraz daha detay verebilir misiniz?")
                    else:
                        answer = validation_result.get("original_response", str(greeting_response))
                else:
                    answer = str(greeting_response)
                
                # final_result'ı dictionary olarak ayarla
                final_result = {
                    "answer": answer,
                    "source": "greeting",
                    "confidence": 0.9
                }
                
                print(f"İşlenmiş selamlama yanıtı: {answer}")
                
            except Exception as e:
                print(f"❌ Selamlama agent hatası: {str(e)}")
                final_result = {
                    "answer": f"Merhaba! Ben MNG Asistan HarbiAI. Size nasıl yardımcı olabilirim?",
                    "source": "fallback",
                    "confidence": 0.9
                }
        else:
            # Tur programı/detayları isteği mi kontrol et
            tour_id = find_tour_id_by_name_or_number(request.lastMessage, input_data, conv_context.last_shown_tours)
            if tour_id:
                print(f"\n✅ Tur ID bulundu: {tour_id}")
                try:
                        # mcp.py'den programı çek
                        program = fetch_tour_price(tour_id, username, password)
                        if program and "Veri alırken bir sorun oluştu" not in program:
                            final_result = {
                                "answer": f"Turun gün gün programı aşağıdadır:\n{program}",
                                "source": "mcp.py",
                                "confidence": 0.95,
                                "tour_id": tour_id
                            }
                        else:
                            final_result = {
                                "answer": "Üzgünüm, bu turun program bilgisi şu anda mevcut değil. Lütfen daha sonra tekrar deneyin.",
                                "source": "mcp.py",
                                "confidence": 0.0
                            }
                except Exception as e:
                        print(f"\n❌ mcp.py'den veri çekilirken hata: {str(e)}")
                        final_result = {
                            "answer": "Üzgünüm, program bilgisi alınırken bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
                            "source": "mcp.py",
                            "confidence": 0.0
                        }
            else:
                # Qdrant ve input_data işlemleri
                print("\n=== QDRANT VE INPUT_DATA İŞLEMLERİ ===")
                print(f"Mevcut input_data sayısı: {len(input_data)}")
                
                # Qdrant'tan yanıt al
                print("\nQdrant sorgusu yapılıyor...")
                qdrant_client = get_qdrant_client()  # Sadece gerektiğinde bağlantı aç
                context = await get_context_for_question(request.lastMessage, conv_context)
                print(f"Qdrant yanıtı: {json.dumps(context, ensure_ascii=False, indent=2)}")
                
                # input_data'yı doldur (eğer boşsa)
                if not input_data and context:
                    print("📝 input_data boş, Qdrant sonuçlarından dolduruluyor...")
                    # context'ten tur verilerini çıkar ve input_data'ya ekle
                    if isinstance(context, dict) and 'results' in context:
                        input_data.extend(context['results'])
                        print(f"✅ {len(context['results'])} tur input_data'ya eklendi")
                    elif isinstance(context, list):
                        input_data.extend(context)
                        print(f"✅ {len(context)} tur input_data'ya eklendi")
                    else:
                        print("⚠️ Context'ten tur verisi çıkarılamadı")
                
                # Router agent çalıştır
                print("\n=== ROUTER AGENT ÇALIŞIYOR ===")
                try:
                    router_result = await query_router_agent.run(
                        chat_history,
                        context=context,
                        input_data=input_data
                    )
                    print(f"Router yanıtı: {json.dumps(router_result, ensure_ascii=False, indent=2)}")
                    
                    # Router sonucunun dictionary olduğundan emin ol
                    if isinstance(router_result, list):
                        print("⚠️ Router agent liste döndürdü, ilk eleman alınıyor...")
                        if router_result:
                            router_result = router_result[0] if isinstance(router_result[0], dict) else {"context": router_result}
                        else:
                            router_result = {"context": []}
                    elif not isinstance(router_result, dict):
                        router_result = {"context": str(router_result)}
                        
                except Exception as e:
                    print(f"❌ Router agent hatası: {str(e)}")
                    router_result = {"context": []}

                # Reply agent çalıştır
                print("\n=== REPLY AGENT ÇALIŞIYOR ===")
                try:
                    final_result = await reply_agent.run(
                        chat_history,
                        context=router_result,
                        input_data=input_data
                    )
                    print(f"Reply agent yanıtı: {final_result}")
                    print(f"Reply agent yanıt tipi: {type(final_result)}")
                    
                    # Eğer final_result liste ise, ilk elemanı al
                    if isinstance(final_result, list):
                        print("⚠️ Reply agent liste döndürdü, ilk eleman alınıyor...")
                        if final_result:
                            if isinstance(final_result[0], dict):
                                final_result = final_result[0]
                            else:
                                final_result = {"answer": str(final_result[0]), "source": "list_first", "confidence": 0.8}
                        else:
                            final_result = {"answer": "Boş yanıt alındı", "source": "empty_list", "confidence": 0.0}
                    elif not isinstance(final_result, dict):
                        # Eğer dictionary değilse, string'e çevir
                        final_result = {"answer": str(final_result), "source": "converted", "confidence": 0.8}
                    
                    print(f"İşlenmiş final_result: {final_result}")
                    
                except Exception as e:
                    print(f"❌ Reply agent hatası: {str(e)}")
                    final_result = {
                        "answer": f"Üzgünüm, yanıt oluşturulurken bir hata oluştu: {str(e)}",
                        "source": "error",
                        "confidence": 0.0
                    }
        
        # Yanıtı işle
        # final_result'ın tanımlı olduğundan emin ol
        if final_result is None:
            print("⚠️ final_result tanımlanmamış, varsayılan yanıt kullanılıyor")
            final_result = {
                "answer": "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
                "source": "fallback",
                "confidence": 0.0
            }
        
        # final_result'ın dictionary olduğundan emin ol
        if not isinstance(final_result, dict):
            print(f"⚠️ final_result dictionary değil, dönüştürülüyor. Tip: {type(final_result)}")
            if isinstance(final_result, list):
                # Eğer liste ise, ilk elemanı al veya tüm listeyi string'e çevir
                if final_result:
                    final_result = {
                        "answer": str(final_result[0]) if final_result[0] else str(final_result),
                        "source": "converted_from_list",
                        "confidence": 0.5
                    }
                else:
                    final_result = {
                        "answer": "Üzgünüm, boş yanıt alındı.",
                        "source": "empty_list",
                        "confidence": 0.0
                    }
            else:
                # Diğer tipler için string'e çevir
                final_result = {
                    "answer": str(final_result) if final_result else "Üzgünüm, bir hata oluştu.",
                    "source": "converted",
                    "confidence": 0.5
                }
            print(f"✅ final_result dönüştürüldü: {final_result}")
        
        answer = final_result.get("answer", "Üzgünüm, bir hata oluştu.")
        print(f"\nİşlenmiş yanıt: {answer}")
        
        # Eğer answer bir dictionary ise, JSON string'e çevir
        if isinstance(answer, dict):
            answer = json.dumps(answer, ensure_ascii=False)
            print(f"\nJSON'a çevrilmiş yanıt: {answer}")
        
        print("\n=== YANIT HAZIRLANIYOR ===")
        final_response = SocialResponse(
            success=True,
            data=ResponseData(
                companyId=request.companyId,
                chatId=request.chatId,
                type="message",
                content=answer
            )
        )
        print(f"Final yanıt: {final_response}")
        print("\n=== İŞLEM TAMAMLANDI ===\n")
        
        return final_response
            
    except Exception as e:
        print(f"\n!!! HATA OLUŞTU !!!")
        print(f"Hata detayı: {str(e)}")
        return SocialResponse(
            success=False,
            errorMessage=str(e)
        )

def get_tour_details_prompt(user_input: str, last_shown_tours: list) -> str:
    """
    Agent'a tur detaylarını sormak için prompt hazırla
    """
    # Son gösterilen turları daha okunabilir formatta hazırla
    formatted_tours = []
    for i, tour in enumerate(last_shown_tours, 1):
        tour_name = tour.get('payload', {}).get('isim', 'İsimsiz Tur')
        formatted_tours.append(f"{i}. {tour_name}")
    
    prompt = f"""
    Kullanıcı şu soruyu sordu: "{user_input}"
    
    Son gösterilen turlar:
    {chr(10).join(formatted_tours)}
    
    Lütfen:
    1. Kullanıcının hangi turun detaylarını istediğini belirle
    2. Bu turun ID'sini bul
    3. Sadece tur ID'sini döndür
    
    Önemli Kurallar:
    - Kullanıcı "X. tur" dediğinde (örn: "2. tur"), X numaralı turu bul
    - Kullanıcı tur ismi verdiğinde, tam eşleşme veya kısmi eşleşme ara
    - Kullanıcı "bu tur" dediğinde, en son konuşulan turu kullan
    - Emin değilsen, kullanıcıdan açıklama iste
    
    Örnek çıktı formatı:
    {{
        "tour_id": "1234"
    }}
    """
    return prompt

def get_agent_response(prompt: str) -> str:
    """
    Agent'dan yanıt al
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen bir tur asistanısın. Kullanıcının hangi turun detaylarını istediğini belirleyip, o turun ID'sini bulmalısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Agent yanıtı alınamadı: {str(e)}")
        return "{}"

def find_tour_id_by_name_or_number(user_input: str, input_data: list, last_shown_tours: list) -> Union[str, None]:
    """
    Kullanıcının belirttiği turun id'sini bul.
    Önce agent ile dene, başarısız olursa manuel aramaya geç.
    """
    print("\n🔍 Tur ID arama başladı...")
    print(f"Kullanıcı girdisi: {user_input}")
    print(f"Son gösterilen tur sayısı: {len(last_shown_tours)}")
    print(f"Toplam tur sayısı: {len(input_data)}")
    
    # Önce agent ile dene
    print("\n🤖 Agent ile tur ID'si aranıyor...")
    prompt = get_tour_details_prompt(user_input, last_shown_tours)
    response = get_agent_response(prompt)
    
    try:
        result = json.loads(response)
        tour_id = result.get("tour_id")
        if tour_id:
            print(f"✅ Agent tur ID'sini buldu: {tour_id}")
            # ID'nin geçerli olduğunu kontrol et
            for tour in input_data:
                if str(tour.get("id", "")) == str(tour_id):
                    return str(tour_id)
            print("❌ Agent'ın bulduğu ID geçersiz, manuel aramaya geçiliyor...")
    except json.JSONDecodeError:
        print("❌ Agent yanıtı geçerli JSON formatında değil")
    
    # Agent başarısız olursa manuel aramaya geç
    print("\n📌 Manuel arama yapılıyor...")
    import re
    
    # Önce sıra numarası ile arama (örn: "2. tur", "2 tur", "2.tur")
    match = re.search(r'(\d+)\.?\s*tur', user_input.lower())
    if match:
        idx = int(match.group(1)) - 1
        print(f"\n📌 Sıra numarası ile arama yapılıyor...")
        print(f"Bulunan sıra numarası: {idx + 1}")
        
        if 0 <= idx < len(last_shown_tours):
            tour = last_shown_tours[idx]
            tour_name = tour.get('payload', {}).get('isim', 'İsimsiz Tur')
            print(f"✅ {idx + 1}. tur bulundu!")
            print(f"Tur adı: {tour_name}")
            
            # Tur adını kullanarak input_data içinden gerçek ID'yi bul
            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id
    
    # Sonra isim ile arama
    print("\n📌 İsim ile arama yapılıyor...")
    print("Son gösterilen turlar içinde aranıyor...")
    for tour in last_shown_tours:
        payload = tour.get("payload", {})
        tour_name = payload.get("isim", "").lower()
        if tour_name and tour_name in user_input.lower():
            print(f"✅ Tur ismi ile bulundu!")
            print(f"Tur adı: {tour_name}")
            
            # Tur adını kullanarak input_data içinden gerçek ID'yi bul
            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id
    
    # Son gösterilen turlarda bulunamazsa, tüm input_data içinde ara
    print("\n📌 Tüm turlar içinde arama yapılıyor...")
    for tour in input_data:
        payload = tour.get("payload", {})
        tour_name = payload.get("isim", "").lower()
        if tour_name and tour_name in user_input.lower():
            tour_id = str(tour.get("id", ""))
            print(f"✅ Tüm turlar içinde bulundu!")
            print(f"Tur adı: {tour_name}")
            print(f"Tur ID: {tour_id}")
            return tour_id
    
    print("\n❌ Tur bulunamadı!")
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

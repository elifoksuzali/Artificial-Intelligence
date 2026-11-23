# logic.py

import json
import logging
from core.core_context import ConversationContext
from models.schema import SocialRequest, SocialResponse, ResponseData
from agent.agent import agent_manager
from mcp.mcp import fetch_tour_price, username, password
from main_logic.main import input_data, find_tour_id_by_name_or_number
from typing import Dict

# Global prompt manager'a erişim için import
from prompts.global_prompt_manager import global_prompt_manager

# Filter search için kullanılacak ana alanlar
FILTER_SEARCH_FIELDS = {"turtipi", "vizesiz", "ulasimtipi", "turkodu", "gecesayisi", "kesinkalkis"}

# Logger
logger = logging.getLogger(__name__)

# Global context sözlüğü (chatId -> ConversationContext)
conversation_contexts: Dict[str, ConversationContext] = {}

def get_user_addressing(firstName: str, gender: str) -> str:
    """Kullanıcı adı ve cinsiyetine göre uygun hitap şeklini döndür"""
    if gender.upper() == "M":
        return f"{firstName} Bey"
    elif gender.upper() == "F":
        return f"{firstName} Hanım"
    else:
        return firstName

async def handle_social_message(request: SocialRequest) -> SocialResponse:
    global input_data  # Global değişken tanımlaması
    try:
        logger.info("\n=== YENİ MESAJ İŞLEME BAŞLADI ===")
        logger.info(f"Gelen mesaj: '{request.lastMessage}'")
        logger.info(f"Mesaj uzunluğu: {len(request.lastMessage)}")
        logger.info(f"Mesaj karakterleri: {[ord(c) for c in request.lastMessage]}")
        logger.info(f"Mesaj hex: {request.lastMessage.encode('utf-8').hex()}")
        
        # Kullanıcı hitap şeklini belirle
        user_addressing = get_user_addressing(request.customer.firstName, request.customer.gender)
        logger.info(f"Kullanıcı hitap şekli: {user_addressing}")

        # Agent'ların durumunu kontrol et ve gerekirse initialize et
        if not agent_manager.is_initialized():
            logger.info("🔧 Agent'lar henüz initialize edilmemiş, şimdi initialize ediliyor...")
            # Global prompt manager'dan prompt'ları al
            if not global_prompt_manager.prompts:
                raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")
            await agent_manager.initialize_agents(global_prompt_manager.prompts)
            logger.info("✅ Agent'lar başarıyla initialize edildi!")
        
        # Agent'ların durumunu kontrol et
        logger.info(f"🔍 Agent durumları:")
        logger.info(f"   question_analyzer_agent: {agent_manager.question_analyzer_agent}")
        logger.info(f"   reply_agent: {agent_manager.reply_agent}")
        logger.info(f"   query_router_agent: {agent_manager.query_router_agent}")
        logger.info(f"   response_validator_agent: {agent_manager.response_validator_agent}")
        logger.info(f"   initialized: {agent_manager.is_initialized()}")
        
        # Eğer agent'lar None ise hata ver
        if not agent_manager.is_initialized():
            raise Exception("Agent'lar henüz initialize edilmemiş!")
        if agent_manager.question_analyzer_agent is None:
            raise Exception("Question Analyzer Agent initialize edilmemiş!")
        if agent_manager.reply_agent is None:
            raise Exception("Reply Agent initialize edilmemiş!")
        if agent_manager.query_router_agent is None:
            raise Exception("Query Router Agent initialize edilmemiş!")
        if agent_manager.response_validator_agent is None:
            raise Exception("Response Validator Agent initialize edilmemiş!")

        # Session ID olarak chatId kullan
        session_id = request.chatId
        logger.info(f"Session ID (chatId): {session_id}")
        
        # Session için context oluştur veya mevcut olanı kullan
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = ConversationContext()
            logger.info("Yeni conversation context oluşturuldu")
        else:
            logger.info("Mevcut conversation context kullanılıyor")
        
        conv_context = conversation_contexts[session_id]
        
        # Chat history'yi oluştur
        chat_history = []
        
        # Müşteri bilgilerini ekle
        customer_info = {
            "role": "system",
            "content": f"Kullanıcı Bilgileri:\nAd: {request.customer.firstName}\nSoyad: {request.customer.lastName}\nCinsiyet: {request.customer.gender}\nHitap Şekli: {user_addressing}"
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
        
        # Geçmiş mesaj kontrolü - "az önce ne sordum" gibi sorular için
        if any(keyword in request.lastMessage.lower() for keyword in ["az önce", "ne sordum", "ne dedim", "hatırla"]):
            print("\n🔍 Geçmiş mesaj sorusu tespit edildi!")
            # Son kullanıcı mesajını bul
            last_user_message = None
            for msg in reversed(request.previousMessages):
                if msg.type == "USER":
                    last_user_message = msg.content
                    break
            
            if last_user_message:
                print(f"✅ Son kullanıcı mesajı bulundu: {last_user_message}")
                # Özel yanıt oluştur - dinamik hitap kullan
                answer = f"{user_addressing}, az önce '{last_user_message}' şeklinde bir soru sordunuz."
                final_result = {
                    "answer": answer,
                    "source": "chat_history",
                    "confidence": 0.95
                }
            else:
                final_result = {
                    "answer": "Üzgünüm, önceki mesajınızı bulamadım.",
                    "source": "chat_history",
                    "confidence": 0.0
                }
            
            # Yanıtı işle ve döndür
            answer = final_result.get("answer", "Üzgünüm, bir hata oluştu.")

        else:
            # 4. ADIM: Yazım hatası kontrolü ve düzeltme
            print("\n4. Yazım hatası kontrolü yapılıyor...")
            print(f"   Orijinal soru: '{request.lastMessage}'")
            
            corrected_message = await agent_manager.query_correction_agent.run([{
                "role": "user",
                "content": request.lastMessage
            }])
            corrected_content = corrected_message.get("answer", request.lastMessage)
            
            # Yazım hatası var mı kontrol et
            if corrected_content.lower() != request.lastMessage.lower():
                print(f"4. Yazım hatası var, düzeltme yapılıyor...")
                print(f"   Orijinal: '{request.lastMessage}'")
                print(f"   Düzeltilen: '{corrected_content}'")
                print(f"   Değişiklik: Yazım hatası düzeltildi")
            else:
                print("4. Yazım hatası yok, bir sonraki işleme geçiliyor")
                print(f"   Orijinal: '{request.lastMessage}'")
                print(f"   Düzeltilen: '{corrected_content}' (değişiklik yok)")
            
            # Düzeltilen içeriği chat_history'ye ekle
            if corrected_content != request.lastMessage:
                # Son mesajı düzeltilmiş haliyle güncelle
                chat_history[-1]["content"] = corrected_content
                print(f"   Chat history güncellendi: '{corrected_content}'")
            
            # 5. ADIM: Soru analizi yapılır
            print("\n5. Soru analizi yapılıyor...")
            analysis = await agent_manager.question_analyzer_agent.run(chat_history)
            print(f"✅ Analiz tamamlandı: {analysis}")
            
            # Analiz sonucunu parse et - Question Analyzer Agent JSON formatında döndürüyor
            try:
                # Question Analyzer Agent artık {"answer": "JSON_STRING", "source": "gpt", "confidence": 0.8} formatında döndürüyor
                if isinstance(analysis, dict):
                    analysis_content = analysis.get("answer", "")
                    
                    # answer JSON string olarak geliyor, parse et
                    if isinstance(analysis_content, str) and analysis_content.startswith('{') and analysis_content.endswith('}'):
                        try:
                            parsed_answer = json.loads(analysis_content)
                            analysis_type = parsed_answer.get("type", "unknown")
                            print(f"✅ Analiz tipi belirlendi: {analysis_type}")
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON parse hatası: {e}")
                            analysis_type = "unknown"
                    else:
                        print(f"⚠️ Answer JSON formatında değil: {analysis_content}")
                        analysis_type = "unknown"
                else:
                    print(f"⚠️ Analysis dict değil: {type(analysis)}")
                    analysis_type = "unknown"
                    
            except Exception as e:
                print(f"❌ Analiz parse hatası: {str(e)}")
                analysis_type = "unknown"
            
            # 6. ADIM: Soru tipine göre agent seçimi
            if analysis_type == "greeting":
                print("\n5.1. Greeting agent cevap hazırlıyor")
                final_result = await agent_manager.greeting_agent.run(chat_history)
                print(f"Selamlama yanıtı: {final_result}")
                
                # Yanıtı kontrol et ve düzelt
                validation_result = await agent_manager.response_validator_agent.run([
                    {"role": "system", "content": "Yanıt kontrolü yapılacak"},
                    {"role": "user", "content": str(final_result)}
                ])
                
                # Eğer yanıt geçerli değilse, düzeltilmiş yanıtı kullan
                if isinstance(validation_result, dict):
                    if not validation_result.get("is_valid", True):
                        answer = validation_result.get("corrected_response", f"Üzgünüm {user_addressing}, mesajınızı tam olarak anlayamadım. Size daha iyi yardımcı olabilmem için biraz daha detay verebilir misiniz?")
                    else:
                        answer = validation_result.get("original_response", str(final_result))
                else:
                    answer = str(final_result)
                
                print(f"İşlenmiş selamlama yanıtı: {answer}")
                
            elif analysis_type =="tour_list": # in ["tour_list", "tour_info"]:
                print(f"\n5.1. Sorulan sorunun alanları tespit edildi")
                
                # 5.2. Tur analizi yaparak field'ları çıkar
                print("\n5.2. Tur analizi yapılıyor...")
                tour_analysis = await agent_manager.tour_analysis_agent.run([{
                    "role": "user",
                    "content": corrected_content
                }])
                tour_analysis_content = tour_analysis.get("answer", "")
                print(f"✅ Tur analizi: {tour_analysis_content}")
                
                # Tur analizi sonucunu parse et
                tour_fields = {}
                if tour_analysis_content:
                    try:
                        # Eğer JSON formatında ise parse et
                        if tour_analysis_content.startswith('{') and tour_analysis_content.endswith('}'):
                            # JSON string'i temizle ve parse et
                            cleaned_content = tour_analysis_content.replace("'", '"')
                            tour_fields = json.loads(cleaned_content)
                        else:
                            # Eğer string formatında ise, basit parsing yap
                            lines = tour_analysis_content.split('\n')
                            for line in lines:
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    tour_fields[key.strip()] = value.strip()
                    except Exception as e:
                        print(f"⚠️ Tur analizi parse hatası: {e}")
                        # Fallback: String parsing dene
                        try:
                            # Sıngle quote'ları double quote'a çevir
                            cleaned_content = tour_analysis_content.replace("'", '"')
                            tour_fields = json.loads(cleaned_content)
                            print(f"✅ Fallback parsing başarılı: {tour_fields}")
                        except Exception as e2:
                            print(f"❌ Fallback parsing de başarısız: {e2}")
                            # Son çare: Manuel parsing
                            tour_fields = {}
                            if 'gecesayisi' in tour_analysis_content:
                                import re
                                match = re.search(r"gecesayisi['\"]?\s*:\s*['\"]?(\d+)", tour_analysis_content)
                                if match:
                                    tour_fields['gecesayisi'] = match.group(1)
                            if 'turtipi' in tour_analysis_content:
                                match = re.search(r"turtipi['\"]?\s*:\s*['\"]?([^'\"]+)", tour_analysis_content)
                                if match:
                                    tour_fields['turtipi'] = match.group(1)
                            print(f"🔧 Manuel parsing sonucu: {tour_fields}")
                
                print(f"✅ Çıkarılan alanlar: {tour_fields}")
                
                # 5.3. input_data kontrolü
                print("\n5.3. input_data veri var mı kontrol edildi")
                if not input_data or len(input_data) == 0:
                    print("5.3. input_data boşsa Qdrant istek atılacak")
                    
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
                    
                    # Qdrant ve input_data işlemleri
                    print("\n=== QDRANT VE INPUT_DATA İŞLEMLERİ ===")
                    print(f"Mevcut input_data sayısı: {len(input_data)}")
                    
                    # Qdrant'tan yanıt al - tour_fields kullanarak filter search yap
                    print("\nQdrant sorgusu yapılıyor...")
                    
                    # main_logic.main'den get_context fonksiyonunu import et ve kullan
                    from main_logic.main import get_context
                    
                    # tour_fields'i filter search için hazırla ve FILTER_SEARCH_FIELDS kontrolü yap
                    filter_fields = []
                    non_filter_fields = []
                    
                    for key, value in tour_fields.items():
                        if key in FILTER_SEARCH_FIELDS:
                            filter_fields.append({"field": key, "value": value})
                        else:
                            non_filter_fields.append({"field": key, "value": value})
                    
                    # Eğer FILTER_SEARCH_FIELDS'de olmayan alanlar varsa multivector search kullan
                    if non_filter_fields:
                        print(f"⚠️ FILTER_SEARCH_FIELDS'de olmayan alanlar tespit edildi: {[f['field'] for f in non_filter_fields]}")
                        print(f"✅ Multivector search kullanılıyor - Soru: '{corrected_content}'")
                        
                        # Multivector search yap
                        from search.multivector_search import search_tours
                        search_result = search_tours(corrected_content, top_k=10)
                        
                        if search_result.get("success") and search_result.get("data"):
                            input_data = search_result["data"]
                            print(f"✅ {len(input_data)} sonuç input_data'ya aktarıldı")
                            
                            # Context oluştur
                            context = {
                                "detail_field": "multivector_search",
                                "tour_category": "multivector",
                                "type": "tour_list",
                                "shown_result_ids": list(conv_context.shown_result_ids),
                                "last_shown_tours": conv_context.last_shown_tours,
                                "results": search_result["data"]
                            }
                        else:
                            print("❌ Multivector search sonuç vermedi")
                            context = {"results": []}
                    
                    # Eğer sadece FILTER_SEARCH_FIELDS'de olan alanlar varsa filter search kullan
                    elif filter_fields:
                        print(f"✅ Filter search kullanılıyor - Alanlar: {[f['field'] for f in filter_fields]}")
                        # Filter search yap
                        from search.filter_search import create_filter_only_search, search_structured
                        
                        filter_conditions = create_filter_only_search(filter_fields)
                        qdrant_params = {
                            "filter": filter_conditions,
                            "limit": 5,
                            "with_payload": True,
                            "with_vector": False
                        }
                        
                        search_result = search_structured(qdrant_params, False)
                        
                        if search_result.get("success") and search_result.get("data"):
                            input_data = search_result["data"]
                            print(f"✅ {len(input_data)} sonuç input_data'ya aktarıldı")
                            
                            # tour_list analiz tipi için determine_agent_and_extract kullan
                            if analysis_type == "tour_list":
                                print("\n🔍 Tour list analiz tipi tespit edildi, determine_agent_and_extract çalıştırılıyor...")
                                
                                # Sadece tur isimlerini çıkar
                                result_tour_list = []
                                for tour in input_data:
                                    if tour.get("payload", {}).get("isim"):
                                        result_tour_list.append({
                                            "id": tour.get("id"),
                                            "isim": tour.get("payload", {}).get("isim")
                                        })
                                
                                print(f"✅ {len(result_tour_list)} tur ismi çıkarıldı")
                                print("📋 Tur isimleri:", [tour["isim"] for tour in result_tour_list])
                                
                                # Context'i sadece tur isimleri ile güncelle
                                context = {
                                    "detail_field": "filter_search",
                                    "tour_category": "filter",
                                    "type": "tour_list",
                                    "shown_result_ids": list(conv_context.shown_result_ids),
                                    "last_shown_tours": conv_context.last_shown_tours,
                                    "results": result_tour_list  # Sadece isim ve ID
                                }
                            else:
                                # Diğer analiz tipleri için normal context
                                context = {
                                    "detail_field": "filter_search",
                                    "tour_category": "filter",
                                    "type": "tour_list",
                                    "shown_result_ids": list(conv_context.shown_result_ids),
                                    "last_shown_tours": conv_context.last_shown_tours,
                                    "results": search_result["data"]
                                }
                        else:
                            print("❌ Filter search sonuç vermedi")
                            context = {"results": []}
                    
                    # Hiç alan bulunamadıysa multivector search kullan
                    else:
                        print("⚠️ Hiç alan bulunamadı, multivector search kullanılıyor")
                        from search.multivector_search import search_tours
                        search_result = search_tours(corrected_content, top_k=10)
                        
                        if search_result.get("success") and search_result.get("data"):
                            input_data = search_result["data"]
                            print(f"✅ {len(input_data)} sonuç input_data'ya aktarıldı")
                            
                            # Context oluştur
                            context = {
                                "detail_field": "multivector_search",
                                "tour_category": "multivector",
                                "type": "tour_list",
                                "shown_result_ids": list(conv_context.shown_result_ids),
                                "last_shown_tours": conv_context.last_shown_tours,
                                "results": search_result["data"]
                            }
                        else:
                            print("❌ Multivector search sonuç vermedi")
                            context = {"results": []}
                    
                    print(f"Qdrant yanıtı: {json.dumps(context, ensure_ascii=False, indent=2)}")
                    
                    # Determine agent and extract bilgileri - Önce çalıştır
                    print("\n=== AGENT DETERMINATION ÇALIŞIYOR ===")
                    from main_logic.main import determine_agent_and_extract
                    
                    agent_name, extracted_items = await determine_agent_and_extract(
                        corrected_content, 
                        input_data, 
                        chat_history
                    )
                    print(f"Seçilen agent: {agent_name}")
                    print(f"Çıkarılan öğeler: {extracted_items}")
                    
                    # Context'e agent bilgisini ekle
                    context["selected_agent"] = agent_name
                    context["extracted_items"] = extracted_items
                    
                    # Router agent çalıştır - Elde edilen veriyi gönder
                    print("\n=== ROUTER AGENT ÇALIŞIYOR ===")
                    router_result = await agent_manager.query_router_agent.run(
                        chat_history,
                        context=context,
                        input_data=extracted_items  # Çıkarılan öğeleri gönder
                    )
                    print(f"Router yanıtı: {json.dumps(router_result, ensure_ascii=False, indent=2)}")
                    
                    # Reply agent çalıştır
                    print("\n=== REPLY AGENT ÇALIŞIYOR ===")
                    final_result = await agent_manager.reply_agent.run(
                        chat_history,
                        context=router_result,
                        input_data=extracted_items  # Çıkarılan öğeleri gönder
                    )
                    print(f"Final yanıt: {json.dumps(final_result, ensure_ascii=False, indent=2)}")
                    
                else:
                    print("5.3. input_data veri var, mevcut veri kullanılıyor")
                    # Mevcut input_data ile işlem yap
                    final_result = {
                        "answer": f"{user_addressing}, mevcut tur verilerinizle size yardımcı olabilirim.",
                        "source": "input_data",
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
                    print("\n❌ Tur ID bulunamadı")
                    # Qdrant ve input_data işlemleri
                    print("\n=== QDRANT VE INPUT_DATA İŞLEMLERİ ===")
                    print(f"Mevcut input_data sayısı: {len(input_data)}")
                    
                    # Qdrant'tan yanıt al - tour_fields kullanarak filter search yap
                    print("\nQdrant sorgusu yapılıyor...")
                    
                    # tour_fields'i filter search için hazırla
                    filter_fields = []
                    for key, value in tour_fields.items():
                        if key in ["turtipi", "vizesiz", "ulasimtipi", "turkodu", "gecesayisi", "kesinkalkis"]:
                            filter_fields.append({"field": key, "value": value})
                    
                    if filter_fields:
                        print(f"✅ Filter search kullanılıyor - Alanlar: {[f['field'] for f in filter_fields]}")
                        # Filter search yap
                        from search.filter_search import create_filter_only_search, search_structured
                        
                        filter_conditions = create_filter_only_search(filter_fields)
                        qdrant_params = {
                            "filter": filter_conditions,
                            "limit": 5,
                            "with_payload": True,
                            "with_vector": False
                        }
                        
                        search_result = search_structured(qdrant_params, False)
                        
                        if search_result.get("success") and search_result.get("data"):
                            input_data = search_result["data"]
                            print(f"✅ {len(input_data)} sonuç input_data'ya aktarıldı")
                            
                            # Context oluştur
                            context = {
                                "detail_field": "filter_search",
                                "tour_category": "filter",
                               "type": "tour_list",
                                "shown_result_ids": list(conv_context.shown_result_ids),
                                "last_shown_tours": conv_context.last_shown_tours,
                                "results": search_result["data"]
                            }
                        else:
                            print("❌ Filter search sonuç vermedi")
                            context = {"results": []}
                    else:
                        print("✅ Multivector search kullanılıyor")
                        # Multivector search yap
                        from search.multivector_search import search_tours
                        search_result = search_tours(corrected_content, top_k=10)
                        
                        if search_result.get("success") and search_result.get("data"):
                            input_data = search_result["data"]
                            print(f"✅ {len(input_data)} sonuç input_data'ya aktarıldı")
                            
                            # Context oluştur
                            context = {
                                "detail_field": "multivector_search",
                                "tour_category": "multivector",
                                "type": "tour_list",
                                "shown_result_ids": list(conv_context.shown_result_ids),
                                "last_shown_tours": conv_context.last_shown_tours,
                                "results": search_result["data"]
                            }
                        else:
                            print("❌ Multivector search sonuç vermedi")
                            context = {"results": []}
                    
                    #print(f"Qdrant yanıtı: {json.dumps(context, ensure_ascii=False, indent=2)}")
                    
                    # Determine agent and extract bilgileri - Önce çalıştır
                    print("\n=== AGENT DETERMINATION ÇALIŞIYOR ===")
                    from main_logic.main import determine_agent_and_extract
                    
                    agent_name, extracted_items = await determine_agent_and_extract(
                        corrected_content, 
                        input_data, 
                        chat_history
                    )
                    print(f"Seçilen agent: {agent_name}")
                    print(f"Çıkarılan öğeler: {extracted_items}")
                    
                    # Context'e agent bilgisini ekle
                    context["selected_agent"] = agent_name
                    context["extracted_items"] = extracted_items
                    
                    # Router agent çalıştır - Elde edilen veriyi gönder
                    print("\n=== ROUTER AGENT ÇALIŞIYOR ===")
                    router_result = await agent_manager.query_router_agent.run(
                        chat_history,
                        context=context,
                        input_data=extracted_items  # Çıkarılan öğeleri gönder
                    )
                    print(f"Router yanıtı: {json.dumps(router_result, ensure_ascii=False, indent=2)}")
                    
                    # Reply agent çalıştır
                    print("\n=== REPLY AGENT ÇALIŞIYOR ===")
                    final_result = await agent_manager.reply_agent.run(
                        chat_history,
                        context=router_result,
                        input_data=extracted_items  # Çıkarılan öğeleri gönder
                    )
                    print(f"Final yanıt: {json.dumps(final_result, ensure_ascii=False, indent=2)}")
            
        # Yanıtı işle
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
        print(f"❌ Mesaj işleme hatası: {str(e)}")
        return SocialResponse(
            success=False,
            errorMessage=f"Mesaj işlenirken hata oluştu: {str(e)}"
        )
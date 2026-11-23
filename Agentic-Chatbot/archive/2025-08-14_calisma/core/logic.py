# logic.py

import json
from core.context import ConversationContext
from models.schema import SocialRequest, SocialResponse, ResponseData
from agent.agent import agent_manager
from mcp.mcp import fetch_tour_price, username, password
from main.main import get_context_for_question, input_data, find_tour_id_by_name_or_number
from typing import Dict

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
    try:
        print("\n=== YENİ MESAJ İŞLEME BAŞLADI ===")
        print(f"Gelen mesaj: {request.lastMessage}")

        # Kullanıcı hitap şeklini belirle
        user_addressing = get_user_addressing(request.customer.firstName, request.customer.gender)
        print(f"Kullanıcı hitap şekli: {user_addressing}")

        # Agent'ların durumunu kontrol et ve gerekirse initialize et
        if not agent_manager.is_initialized():
            print("🔧 Agent'lar henüz initialize edilmemiş, şimdi initialize ediliyor...")
            await agent_manager.initialize_agents()
            print("✅ Agent'lar başarıyla initialize edildi!")
        
        # Agent'ların durumunu kontrol et
        print(f"🔍 Agent durumları:")
        print(f"   question_analyzer_agent: {agent_manager.question_analyzer_agent}")
        print(f"   reply_agent: {agent_manager.reply_agent}")
        print(f"   query_router_agent: {agent_manager.query_router_agent}")
        print(f"   response_validator_agent: {agent_manager.response_validator_agent}")
        print(f"   initialized: {agent_manager.is_initialized()}")
        
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
            print("\n🔍 Soru analizi yapılıyor...")
            analysis = await agent_manager.question_analyzer_agent.run(chat_history)
            print(f"✅ Analiz tamamlandı: {analysis}")
            
            # Analiz sonucunu JSON olarak parse et - HATA DÜZELTİLDİ
            try:
                if isinstance(analysis, str):
                    # Eğer analysis string ise, JSON olarak parse etmeye çalış
                    try:
                        analysis = json.loads(analysis)
                    except json.JSONDecodeError:
                        # JSON parse edilemezse, basit bir dict oluştur
                        analysis = {"answer": {"type": "unknown"}}
                
                # analysis artık dict olduğundan emin ol
                if isinstance(analysis, dict):
                    analysis_type = analysis.get("answer", {}).get("type", "unknown")
                else:
                    analysis_type = "unknown"
                    
            except Exception as e:
                print(f"❌ Analiz parse hatası: {str(e)}")
                analysis_type = "unknown"
            
            # Eğer selamlama ise direkt reply_agent'ı kullan
            if analysis_type == "greeting":
                print("\n🤖 Selamlama yanıtı hazırlanıyor...")
                final_result = await agent_manager.reply_agent.run(chat_history)
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
                    
                    # Qdrant'tan yanıt al
                    print("\nQdrant sorgusu yapılıyor...")
                    context = await get_context_for_question(request.lastMessage, conv_context)
                    print(f"Qdrant yanıtı: {json.dumps(context, ensure_ascii=False, indent=2)}")
                    
                    # Router agent çalıştır
                    print("\n=== ROUTER AGENT ÇALIŞIYOR ===")
                    router_result = await agent_manager.query_router_agent.run(
                        chat_history,
                        context=context,
                        input_data=input_data
                    )
                    print(f"Router yanıtı: {json.dumps(router_result, ensure_ascii=False, indent=2)}")
                    
                    # Reply agent çalıştır
                    print("\n=== REPLY AGENT ÇALIŞIYOR ===")
                    final_result = await agent_manager.reply_agent.run(
                        chat_history,
                        context=router_result,
                        input_data=input_data
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
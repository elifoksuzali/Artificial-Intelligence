import os
import openai
import time
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

ASSISTANT_ID = "asst_DEMO000000000000000000000"
tour_id = None  # Asistanın hatırlayacağı tur ID'si için başlangıç değeri

def get_user_input():
    user_input = input("Lütfen sorunu girin: ").strip()
    while not user_input:
        print("Boş mesaj gönderilemez. Lütfen bir sorun girin.")
        user_input = input("Lütfen sorunu girin: ").strip()
    return user_input

def collect_price_details():
    tarih = input("Tarih (GG.AA.YYYY): ")
    odasayisi = input("Oda Sayısı: ")
    oda1yetiskin = input("Yetişkin Sayısı: ")
    oda1cocuk1 = input("Çocuk Yaşı 1 (Boş geçebilirsiniz): ") or "0"
    oda1cocuk2 = input("Çocuk Yaşı 2 (Boş geçebilirsiniz): ") or "0"
    
    return {
        "tarih": tarih or "0",
        "odasayisi": odasayisi or "0",
        "oda1yetiskin": oda1yetiskin or "0",
        "oda1cocuk1": oda1cocuk1,
        "oda1cocuk2": oda1cocuk2
    }

def main():
    global tour_id
    
    # Yeni bir thread oluşturun
    thread = openai.beta.threads.create()
    THREAD_ID = thread.id
    print(f"👉 Thread oluşturuldu: {thread.id}")
    
    while True:
        user_query = get_user_input()
        if user_query.lower() == 'exit':
            print("Chat sonlandırıldı. İyi günler.")
            break

        # Mesajı thread'e ekleyin
        my_thread_message = openai.beta.threads.messages.create(
            thread_id=THREAD_ID,
            role="user",
            content=user_query
        )

        # Asistanı çalıştırın
        run = openai.beta.threads.runs.create(thread_id=THREAD_ID, assistant_id=ASSISTANT_ID)
        print(f"👉 Run oluşturuldu: {run.id}")

        # Cevabı bekleyin ve alın
        while run.status in ["queued", "in_progress"]:
            keep_retrieve_run = openai.beta.threads.runs.retrieve(
                thread_id=THREAD_ID,
                run_id=run.id
            )
            print(f"Run durumu: {keep_retrieve_run.status}")

            if keep_retrieve_run.status == "completed":
                print("\n")

                all_messages = openai.beta.threads.messages.list(
                    thread_id=THREAD_ID
                )

                assistant_response = all_messages.data[0].content[0].text.value
                print("--------------------------------------------------------------------------")
                
                if "fiyat" in user_query.lower():
                    # Fiyatla ilgili bir soruysa, önce tur ID'sini hatırla
                    if tour_id is None:
                        # Asistanın cevabında tur ID'sini yakala
                        # (Bu örnekte, cevaptan bir şekilde tur ID'si çıkarıldığını varsayıyoruz)
                        # Gerçek bir sistemde, bu ID'yi spesifik bir function call'dan veya JSON cevabından alırsınız
                        tour_id = "asistanın_bulacağı_tur_id"  # Burada uygun bir şekilde tur ID'sini belirleyin
                        print(f"🔍 Hatırlanan Tur ID: {tour_id}")

                    # Fiyat bilgileri için kullanıcıdan gerekli parametreleri al
                    price_details = collect_price_details()

                    # JSON formatında verileri oluştur
                    post_data = {
                        "key": "aisorgu",
                        "action": "tour-price",
                        "data[id]": tour_id,
                        "data[tarih]": price_details["tarih"],
                        "data[odasayisi]": price_details["odasayisi"],
                        "data[oda1yetiskin]": price_details["oda1yetiskin"],
                        "data[oda1cocuk1]": price_details["oda1cocuk1"],
                        "data[oda1cocuk2]": price_details["oda1cocuk2"]
                    }
                    print(post_data)

                    # Burada post request yapılabilir veya function call çağrılabilir
                    # response = requests.post(SERVICE_URL, data=post_data)

                    print("💬 Fiyat bilgileri işleme alındı.")
                else:
                    # Fiyatla ilgili değilse normal yanıt ver
                    print(f"💬 User: {my_thread_message.content[0].text.value}")
                    print(f"💬 Asistan: {assistant_response}")

                break
            elif keep_retrieve_run.status in ["queued", "in_progress"]:
                pass
            else:
                print(f"Run durumu: {keep_retrieve_run.status}\n")
                break

if __name__ == "__main__":
    main()

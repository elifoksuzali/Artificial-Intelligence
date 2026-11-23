import os
import time
from dotenv import load_dotenv
import openai
from rich.console import Console
import json

console = Console()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Geçici değişkenler
selected_tour = None
tour_list = []

# 1- upload file
my_file = openai.files.create(
    file=open("C:/Users/info/tmps/git/guncel_dosya2.json", "rb"),
    purpose="assistants"
)
file_ids= my_file.id
print(f"File id: {file_ids}\n")

tool_list = {
    "type": "function",
    "function": {
        "name": "get_tour_name_by_name",
        "description": "Kullanıcının seçtiği turun içerik detayını öğrenmek için isim değerini al.",
        "parameters": {
        "type": "object",
        "properties": {
            "isim": {
            "type": "string",
            "description": "Turun ismi."
            }
        },
        "required": ["isim"]
        }
    }
}

# 2- create assistant
my_assistant = openai.beta.assistants.create(
    model="gpt-4o",
    instructions="""
    
    Sen MNG'ye ait müşteri desteği sohbet robotusun. İsmini sorarlarsa veya nasılsın, günün nasıl 
	gidiyor gibi sorular sorulursa, kibarca iyi olduğunu Hitap etmek için önce problemi yaşayan kişiyi 
	veya hangi firmaya ait olduğunu bil müşteri adını kullanarak MNG Asistan Yapay Zeka olduğunu belirterek 
	nasıl yardımcı olabileceğini belirten ifadelerle cevap ver. Json dosyasındaki veriler nasılsa o şekilde 
	listele. Verileri değiştirme başka dile çevirme. Hangi dilde soruluyorsa o dilde cevap ver. Amacın, sana 
	sağlanan bilgiler doğrultusunda müşterilere yardımcı olmak. Kullanıcı gitmek istediği şehri, ülkeyi veya
	kıtayı belirtiyorsa bu bölgeye ait turların [isim] ile birlikte [id] değerlerinin listesini ver. Kullanıcı 
	"örneğin: Paris" gitmek istediğini belirttiğinde Paris içeriği olan tur programlarının [isim] listesini ver.
	Müşteriye hangi turu seçmek istediğini belirtmesini söyle. Sağlanan bilgiler dışında bilgi verme. Kullanıcının
	vermiş olduğu bilgiler doğrultusunda json veri setinden örnekleri sun. Bu bir json dosyasıdır. Dosya okuma işlemi 
	yaparken json olduğunu unutma. 
    Json dosyasında herhangi bir şey araştırırken asla  kullanıcın girdiği kelimeyi bölme bütün halde sorgula.
	Json içinde bilgi sorgularken büyük küçük harf, ilk kelimesi ile başlayacak  şekilde bütün kombinasyonlarını
	dikkat ederek turlara ait bilgileri araştır.
    Tur bölgesi veya otel, gemi, konum bilgisi giriyorsa [turKategori] altında [kategori_isim] benzeyen turları 
	[turKategori] altında [puan] değerlerini dikkate alarak büyükten küçüğe doğru sıralayarak listele.  
    Kullanıcı vizesiz veya vizeli turlar istiyorsa; Vizesiz tur için [vizesiz] "1" ve   [turtipi] yurt dışı 
	olan ve turları [turKategori] altında [puan] değerlerine göre büyükten küçüğe doğru sıralayarak listele. 
	Turları [ulasimtipi] göre turlarının [isim] ile birlikte [id] değerlerinin listesini ver. 
    Turları ilk olarak en fazla 10 örnek tur listesi ver daha fazlası için kullanıcıya "Dilerseniz seçtiğiniz
	tur hakkında daha fazla detay verebilir veya sizlere daha fazla tur programları sunabilirim." de.
    Turun ulaşımı soruluyorsa [ulasim] bilgisini ver.  Turun ulaşım tipi soruluyorsa [ulasimtipi] bilgisini ver.
    Kullanıcı detayları görmek istiyorum demeden asla turun [isim] ile birlikte [id] değerlerinin bilgisi dışında
	bilgi verme. Turun detaylarını görmek istiyorsa özet olarak tur detaylarından bahset.
    Json dosyasında olmayan bir bilgi istiyorsa  json dosyasındaki [url] kolonundaki linki ver  ve "Daha fazla 
	detay için bu linki ziyaret edebilir veya doğrudan bizimle iletişime geçebilirsiniz" de. 
    [url] linkini direk ver asla json dosyasındaki [url] linki bozma.
    Json dosyasında olmayan kolonlara göre araştırma asla yapma. 
    Kullanıcı istemedikçe detay verme. Cümleyi aynen al değişiklik yapma. 
    Konuşma geçmişini dikkate alarak cevapları üret. Geçmişte sorulan içeriklerine uygun konuşmayı devam ettir. 
    Turları [turKategori] altında [puan] değerlerine göre sırala. Listeden sırasına belirterek tur hakkında 
	işlem yapıyorsa doğru şekilde turun [isim] ile birlikte [id] değerlerinin listesini bilgisini ver.
    Kullanıcı tur hakkında detay isterse get_tour_name_by_name çalıştır
	ve kullanıcıdan turun [isim] al. 
    Müşteri detay sormadığı sürece sadece turlara ait  [isim] başlıklarını ver.""",
    name="MNG Test Assistant Elif",
    tools=[tool_list],
    tool_resources={"code_interpreter": {"file_ids": [file_ids]}}
)

print(f"Assistant id: {my_assistant.id}\n")

# 3- Create a new thread
my_thread = openai.beta.threads.create()
print(f"Thread id: {my_thread.id}")

# 4- Add message to thread
my_thread_message = openai.beta.threads.messages.create(
    thread_id=my_thread.id,
    role="user",
    content="yurt dışı turlarının listesini ver",
)
print(f"Message: {my_thread_message}\n")

# 5- Run assistant
my_run = openai.beta.threads.runs.create(
    thread_id=my_thread.id,
    assistant_id=my_assistant.id
)
print(f"Run: {my_run}\n")

while my_run.status in ["queued", "in_progress"]:
    keep_retrieve_run = openai.beta.threads.runs.retrieve(
        thread_id=my_thread.id,
        run_id=my_run.id
    )
    print(f"Run status: {keep_retrieve_run.status}")

    if keep_retrieve_run.status == "completed":
        print("\n")

        all_messages = openai.beta.threads.messages.list(
            thread_id=my_thread.id
        )

        # Cevabın içeriğini kontrol et ve yazdır
        response_text = all_messages.data[0].content[0].text.value
        print(f"Cevap Metni: {response_text}")

        # JSON ayrıştırmasını dene
        try:
            response_content = json.loads(response_text)
            tour_list = response_content["turlar"]

            # Kullanıcıya gösterilecek isimler ve ID'ler
            tour_names_and_ids = [{"isim": tour["isim"], "id": tour["id"]} for tour in tour_list]
            print(f"💬 Asistan: {json.dumps(tour_names_and_ids, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError as e:
            print("JSON ayrıştırma hatası:", e)

        break
    elif keep_retrieve_run.status in ["queued", "in_progress"]:
        time.sleep(5)  # 5 saniye bekle
        pass
    else:
        print(f"Run status: {keep_retrieve_run.status}\n")
        break

# Kullanıcının seçtiği turu al ve geçici değişkende sakla
selected_tour_name = input("Lütfen bir tur seçin: ")
selected_tour = next((tour for tour in tour_list if tour["isim"] == selected_tour_name), None)

if selected_tour:
    print(f"Seçilen tur: {json.dumps(selected_tour, indent=2, ensure_ascii=False)}")
else:
    print("Seçilen tur bulunamadı.")

# Bu noktada selected_tour değişkeni ile işlem yapabilirsiniz

# 6- Define function to retrieve tour details
def get_tour_details_by_name(name):
    return next((tour for tour in tour_list if tour["isim"] == name), None)

# Kullanıcı turun içeriği ile ilgili daha fazla bilgi almak istediğinde bu fonksiyon çalışacak
selected_tour_details = get_tour_details_by_name(selected_tour_name)

if selected_tour_details:
    print(f"Tur Detayları: {json.dumps(selected_tour_details, indent=2, ensure_ascii=False)}")
else:
    print("Tur detayları bulunamadı.")

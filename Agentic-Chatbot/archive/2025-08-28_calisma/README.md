# 🚀 AI Destekli Tur Arama ve Sohbet Sistemi

Bu proje, OpenAI GPT-4 entegrasyonu ile çoklu agent sistemi kullanarak tur arama ve sohbet hizmeti sunan bir API'dir.

## 🏗️ **Mimari Yapı**

### **Ana Bileşenler**
- **FastAPI** tabanlı REST API
- **15 AI Agent** (soru analizi, yanıt, routing, doğrulama, vb.)
- **Qdrant** vector database ile semantik arama
- **MySQL** veritabanı entegrasyonu
- **MCP (Model Context Protocol)** ile harici tur verisi çekme

### **Agent Sırası**
1. **Query Correction Agent** - Yazım hatası düzeltme
2. **Question Analyzer Agent** - Soru tipini belirleme
3. **Tour Analysis Agent** - Tur alanlarını çıkarma
4. **Search Method Selection** - Filter vs Multivector seçimi
5. **Qdrant Search** - Veritabanı sorgusu
6. **Query Router Agent** - Sorgu yönlendirme
7. **Reply Agent** - Yanıt üretimi
8. **Response Validator Agent** - Yanıt doğrulama

## 🔍 **Search Yöntemleri**

### **Filter Search**
- **FILTER_SEARCH_FIELDS**: `["turtipi", "vizesiz", "ulasimtipi", "turkodu", "gecesayisi", "kesinkalkis"]`
- Yapısal arama (MUST/SHOULD koşulları)
- Kesin eşleşme için kullanılır

### **Multivector Search**
- Semantik arama
- Birden fazla vector field ile
- Anlam tabanlı sonuçlar

## 📁 **Dosya Yapısı**

```
├── main/
│   └── main.py              # Ana mantık ve search yöntemi seçimi
├── core/
│   ├── logic.py             # Ana mesaj işleme pipeline'ı
│   └── context.py           # Konuşma context yönetimi
├── agent/
│   ├── agent.py             # Agent manager
│   ├── simpleAgent.py       # Temel agent implementasyonu
│   └── ConversationContext.py # Konuşma context sınıfı
├── search/
│   ├── filter_search.py     # Yapısal arama
│   └── multivector_search.py # Semantik arama
├── qdrant/
│   └── qdrantDb.py         # Qdrant veritabanı işlemleri
├── api/
│   └── api.py              # FastAPI endpoint'leri
├── prompts/
│   ├── handle_db.py        # Prompt yönetimi
│   └── global_prompt_manager.py # Global prompt manager
└── config.py                # Konfigürasyon
```

## 🚀 **Kullanım**

### **API Endpoint**
```bash
POST /social-chatbot/message
```

### **Request Format**
```json
{
    "parametre": "pro",
    "soru": "3 günlük yurt içi turları neler"
}
```

### **Response Format**
```json
{
    "success": true,
    "data": {
        "companyId": "company_id",
        "chatId": "chat_id",
        "type": "message",
        "content": "Yanıt içeriği"
    }
}
```

## 🔧 **Kurulum**

### **Gereksinimler**
- Python 3.8+
- MySQL
- Qdrant
- OpenAI API Key

### **Environment Variables**
```bash
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_PASSWORD=your_db_password
DB_USER=your_db_user
URL_LINK=your_qdrant_url
```

### **Çalıştırma**
```bash
cd api
python api.py
```

## 📊 **İş Akışı**

1. **Mesaj Alınır** → API endpoint'e POST
2. **Yazım Hatası Düzeltilir** → Query Correction Agent
3. **Soru Analiz Edilir** → Question Analyzer Agent
4. **Tur Alanları Çıkarılır** → Tour Analysis Agent
5. **Search Yöntemi Seçilir** → Filter vs Multivector
6. **Qdrant Sorgusu Yapılır** → Seçilen yöntemle
7. **Yanıt Üretilir** → Reply Agent
8. **Yanıt Doğrulanır** → Response Validator Agent
9. **Sonuç Döndürülür** → JSON response

## 🎯 **Özellikler**

- ✅ **Çoklu Agent Sistemi** - 15 farklı AI agent
- ✅ **Akıllı Search Seçimi** - Alan bazlı yöntem seçimi
- ✅ **Yazım Hatası Düzeltme** - Otomatik düzeltme
- ✅ **Context Yönetimi** - Konuşma geçmişi takibi
- ✅ **Hata Yönetimi** - Kapsamlı hata yakalama
- ✅ **Prompt Yönetimi** - Dinamik prompt yükleme

## 🔒 **Güvenlik**

- IP whitelist kontrolü
- API key yönetimi
- Veritabanı güvenliği

## 📝 **Notlar**

- Agent'lar ilk mesaj geldiğinde initialize edilir
- Prompt'lar veritabanından yüklenir (fallback: manuel)
- Search yöntemi otomatik seçilir
- Context her chat session için ayrı tutulur

## 🐛 **Bilinen Sorunlar**

- `main/main.py` dosyası tamamen yeniden yazıldı
- Eksik fonksiyonlar eklendi
- Agent sırası düzenlendi
- Prompt'lar manuel olarak eklendi

## 📞 **Destek**

Herhangi bir sorun için lütfen iletişime geçin. 
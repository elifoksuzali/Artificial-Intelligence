# 🤖 AI Destekli Tur Arama ve Sohbet Asistanı

OpenAI GPT agent'ları, **Qdrant** vektör veritabanı ve **MySQL** kullanarak tur/seyahat
sorularını yanıtlayan, **FastAPI** tabanlı bir sohbet (chatbot) servisidir. Kullanıcı
sorularını analiz eder, semantik + filtreli arama yapar ve doğal dilde yanıt üretir.

---

## 📁 Depo Yapısı

```
.
├── live_project/      # ⭐ Ana/güncel proje (FastAPI servisi)
├── experiments/       # OpenAI Assistants API ile ilk deneme scriptleri
├── archive/           # Geçmiş çalışma sürümleri (tarih sıralı anlık görüntüler)
│   ├── 2025-05-27_calisma/
│   ├── 2025-06-30_calisma/
│   ├── 2025-08-14_calisma/
│   ├── 2025-08-28_calisma/
│   ├── 2026-03-24_calisma/
│   ├── ollama/        # Ollama entegrasyon denemeleri
│   └── proje/
├── .gitignore
└── README.md
```

Geliştirmeye devam etmek için **`live_project/`** klasörünü kullanın; `archive/` ve
`experiments/` yalnızca geçmiş/referans amaçlıdır.

---

## 🏗️ Mimari (live_project)

- **FastAPI** tabanlı REST API (`api.py`) — CORS ve IP whitelist kontrolü
- **Çoklu GPT agent'ı** (`gpt_agents/`, `agentic_service.py`, `runner_service.py`):
  soru düzeltme, soru analizi, arama yöntemi seçimi, yanıt üretimi, yanıt doğrulama
- **Qdrant** vektör araması (`connection_qdrant.py`, `filter_search.py`, `multivector_search.py`)
- **MySQL** veritabanı bağlantısı (`connection_sql.py`)
- **Araçlar** (`tool_calls.py`): fiyat hesaplama, e-posta (SMTP) gönderimi
- **Promptlar** (`prompts_ollama.py`) ve yanıt şemaları (`response_structure.py`)

### Arama yöntemleri
- **Filter Search:** yapısal/kesin eşleşme (tur tipi, vizesiz, ulaşım tipi, tur kodu vb.)
- **Multivector Search:** anlam tabanlı (semantik) arama

---

## ⚙️ Kurulum

Gereksinim: **Python 3.10+**

```bash
# 1) Depoyu klonla
git clone <repo-url>
cd <repo-adi>/live_project

# 2) Sanal ortam
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3) Bağımlılıklar
pip install fastapi uvicorn openai qdrant-client mysql-connector-python python-dotenv pydantic
# (Hazır liste için archive/proje/requirements.txt ve
#  archive/ollama/ollama_entegretion/requirements.txt dosyalarına da bakabilirsiniz.)
```

---

## 🔑 Ortam Değişkenleri (.env)

Kod, tüm gizli bilgileri `.env` dosyasından okur (`os.getenv`). Başlamak için şablonu kopyalayın:

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

`.env` içine **kendi** değerlerinizi yazın:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-FAKEKEY1234567890abcdefghijklmnop

# Veritabanı (MySQL)
DB_HOST=db.example.com
DB_USER=demo_user
DB_PASSWORD=demoPassw0rd123
DB_NAME=demo_db

# Qdrant
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=qdrant-FAKEKEY-0000000000000000
QDRANT_COLLECTION=demo_collection

# E-posta (SMTP) — tool_calls.py için
SMTP_LOGIN=noreply@example.com
SMTP_PASSWORD=demoSmtpPass123

# Embed modelleri
SPARSE_MODEL=Qdrant/bm25
DENSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=BAAI/bge-reranker-base
```

> `.env` dosyası `.gitignore` sayesinde **repoya gönderilmez**; paylaşılan şablon `.env.example`'dır.

---

## ▶️ Çalıştırma

```bash
cd live_project
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API dokümantasyonu: `http://localhost:8000/docs`

---

## 🧪 experiments/

OpenAI **Assistants API** ile yapılmış ilk deneme scriptleri (`app.py`, `chatbot.py`,
`externalApi.py` vb.). Çalıştırmadan önce:

```bash
export OPENAI_API_KEY="..."   # Windows: set OPENAI_API_KEY=...
python experiments/chatbot.py
```

> Not: Bu scriptlerdeki `ASSISTANT_ID` değerleri placeholder'dır; kendi OpenAI asistan
> kimliğinizle değiştirin.

---

## 📝 Notlar

- `.env` içindeki değerler örnektir; kendi veritabanı, OpenAI ve Qdrant bilgilerinizle güncelleyin.
- `archive/` içindeki sürümler kendi başlarına çalışan bağımsız anlık görüntülerdir; her birinin
  kendi `.env.example` dosyası vardır.

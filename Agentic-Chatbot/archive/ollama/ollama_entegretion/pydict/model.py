from pydantic import BaseModel
from typing import List, Optional


# Pydantic modelleri
class Customer(BaseModel):
    customerId: str
    firstName: str
    lastName: str
    gender: str
    phone: str
    email: str

class Message(BaseModel):
    messageId: str
    date: str
    type: str
    content: str

class AISettings(BaseModel):
    endpoint: str
    prompts: List[str] = []

class ChatRequest(BaseModel):
    parametre: str
    companyId: str
    customer: Customer
    chatId: str
    chatType: str
    previousMessages: List[Message]
    lastMessage: str
    aiSettings: AISettings
    
class QuestionAnalysis(BaseModel):
    question: str
    answer: str
    score: int
    analysis: str

# YENİ MODELLER - Agent'lar için

class QuestionCategory(BaseModel):
    """
    1. Model: Sorunun hangi kategori ile ilgili olduğunu belirler
    """
    category: str  # greeting, tour_list, tour_detail, price
    confidence: float  # 0.0 - 1.0 arası güven skoru
    reasoning: str  # Neden bu kategori seçildiğinin açıklaması
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "tour_list",
                "confidence": 0.95,
                "reasoning": "Kullanıcı 'yurt içi 2 günlük turlar neler' diye sorduğu için tour_list kategorisi seçildi"
            }
        }

class QuestionExtraction(BaseModel):
    """
    Basitleştirilmiş soru çıkarma modeli
    """
    turtipi: Optional[str] = None         # yurt içi, yurt dışı
    geceleme: Optional[str] = None        # Gece sayısı
    konaklama: Optional[str] = None       # Konaklama türü
    ulasim: Optional[str] = None          # Ulaşım türü
    ulasimtipi: Optional[str] = None      # Ulaşım tipi
    ziyaretedilecekyerler: Optional[str] = None  # Ziyaret yerleri
    vizesiz: Optional[bool] = None        # Vize durumu
    kesinkalkis: Optional[bool] = None    # Kesin kalkış durumu
    confidence: float = 0.0               # Güven skoru
    reasoning: str = ""                   # Açıklama
    
    class Config:
        json_schema_extra = {
            "example": {
                "turtipi": "yurt içi",
                "geceleme": "2",
                "ulasimtipi": "otobüslü",
                "confidence": 0.95,
                "reasoning": "Metinden tur tipi ve gece sayısı çıkarıldı"
            }
        }

class AgentTask(BaseModel):
    """
    Agent'ların çalışması için task modeli
    """
    task_id: str
    task_type: str  # "category_analysis", "extraction_analysis"
    input_text: str
    category_result: Optional[QuestionCategory] = None
    extraction_result: Optional[QuestionExtraction] = None
    status: str = "pending"  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "task_type": "category_analysis",
                "input_text": "yurt içi 6 günlük Karadeniz Batum turu var mı",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00Z"
            }
        }

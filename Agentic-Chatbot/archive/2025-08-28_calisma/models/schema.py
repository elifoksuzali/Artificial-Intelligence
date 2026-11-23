# region 
from pydantic import BaseModel,ValidationError
from typing import Dict, Any, List, Optional, Union, cast
# Agent modelleri
class ResearchPaperExtractionID(BaseModel):
    id: int
class ResearchPaperExtractionName(BaseModel):
    isim: str
class ResearchPaperExtractionTurKodu(BaseModel):
    turkodu: str
class ResearchPaperExtractionGeceSayisi(BaseModel):
    #gecesayisi: str
    geceleme: str
class ResearchPaperExtractionKonaklama(BaseModel):
    konaklama: str
class ResearchPaperExtractionUlasim(BaseModel):
    ulasim: str
    # ulasimtipi: str  # Bu alanın her zaman döndürülmesini garanti altına alacağız.
class ResearchPaperExtractionZiyaretEdilecekYerler(BaseModel):
    ziyaretedilecekyerler: str
class ResearchPaperExtractionVizeDurumu(BaseModel):
    vizesiz: str
class ResearchPaperExtractionKesinkalkis(BaseModel):
    kesinkalkis: str
class ResearchPaperExtractionUrl(BaseModel):
    url: str

class Customer(BaseModel):
    customerId: str
    firstName: str
    lastName: str
    gender: str
    phone: str
    email: str
    
class PreviousMessage(BaseModel):
    messageId: str
    date: str
    type: str
    content: str
    
class Prompt(BaseModel):
    type: str
    prompt: str
    
class AISettings(BaseModel):
    endpoint: str
    prompts: List[Prompt]       
    
class SocialRequest(BaseModel):
    parametre: str
    companyId: str
    customer: Customer
    chatId: str
    chatType: str
    previousMessages: List[PreviousMessage]
    lastMessage: str
    aiSettings: AISettings 
    
class ResponseData(BaseModel):
    companyId: str
    chatId: str
    type: str
    content: str
    
class SocialResponse(BaseModel):
    success: bool
    data: Optional[ResponseData] = None
    errorMessage: Optional[str] = None


# endregion
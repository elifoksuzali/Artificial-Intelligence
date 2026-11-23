# region Agent
from pydantic import BaseModel,ValidationError
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
    
    

# endregion
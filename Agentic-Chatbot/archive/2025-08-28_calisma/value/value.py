from typing import List, Dict, Any
from models.schema import ResearchPaperExtractionID, ResearchPaperExtractionName, ResearchPaperExtractionTurKodu, ResearchPaperExtractionGeceSayisi, ResearchPaperExtractionKonaklama, ResearchPaperExtractionUlasim, ResearchPaperExtractionZiyaretEdilecekYerler, ResearchPaperExtractionVizeDurumu, ResearchPaperExtractionKesinkalkis, ResearchPaperExtractionUrl

# Global değişkenler
input_data: List[Dict[str, Any]] = []  # Qdrant'tan gelen veriler burada saklanacak
chat_history: List[Dict[str, str]] = []  # Sohbet geçmişi
last_query: Dict[str, Any] = {}  # Son yapılan sorguyu saklamak için
current_page: int = 0  # Mevcut sayfa numarası
PAGE_SIZE: int = 10  # Her sayfada gösterilecek tur sayısı
shown_result_ids: set = set()  # Daha önce gösterilen sonuçların ID'lerini tutacak set
last_query_context: Dict[str, Any] = {  # Son sorgu bağlamını tutacak
    "query_type": None,  # "yurt içi", "yurt dışı" veya None
    "shown_ids": [],  # Son sorguda gösterilen ID'ler
    "total_results": 0,  # Toplam bulunan sonuç sayısı
    "current_offset": 0  # Mevcut offset
}

agent_map = {
    "id": ResearchPaperExtractionID,
    "isim": ResearchPaperExtractionName,
    "turkodu": ResearchPaperExtractionTurKodu,
    "geceleme": ResearchPaperExtractionGeceSayisi,
    "konaklama": ResearchPaperExtractionKonaklama,
    "ulasim": ResearchPaperExtractionUlasim,
    "ziyaretedilecekyerler": ResearchPaperExtractionZiyaretEdilecekYerler,
    "vizesiz": ResearchPaperExtractionVizeDurumu,
    "kesinkalkis": ResearchPaperExtractionKesinkalkis,
    "url": ResearchPaperExtractionUrl
}
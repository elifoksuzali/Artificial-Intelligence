from typing import List, Dict, Any, Optional
from value.value import input_data
class ConversationContext:
    def __init__(self):
        
        self.selected_tour: Optional[Dict[str, Any]] = None  # Seçilen tur
        self.tour_list: List[Dict[str, Any]] = []        # Mevcut tur listesi
        self.query_type: Optional[str] = None     # Tur tipi (yurt içi/dışı)
        self.tour_category: Optional[str] = None  # Tur kategorisi
        self.transport_type: Optional[str] = None # Ulaşım tipi
        self.last_query: Optional[str] = None     # Son soru
        self.is_more_requested: bool = False  # Daha fazla tur istenip istenmediği
        self.shown_result_ids: set = set()  # Gösterilen sonuçların ID'leri
        self.temp_tours: List[Dict[str, Any]] = []  # Geçici tur listesi
        self.last_shown_tours: List[Dict[str, Any]] = []  # Son gösterilen turlar

    def add_tour(self, tour: Dict[str, Any]) -> None:
        """Turu geçici listeye ekle"""
        tour_id = str(tour.get("id", ""))
        if tour_id and tour_id not in self.shown_result_ids:
            self.temp_tours.append(tour)
            self.shown_result_ids.add(tour_id)

    def get_tour_by_id(self, tour_id: str) -> Optional[Dict[str, Any]]:
        """ID'ye göre tur bilgisini getir"""
        # Önce geçici listede ara
        for tour in self.temp_tours:
            if str(tour.get("id", "")) == tour_id:
                return tour
        # Sonra input_data'da ara
        for tour in input_data:
            if str(tour.get("id", "")) == tour_id:
                return tour
        return None

    def get_tour_by_number(self, number: int) -> Optional[Dict[str, Any]]:
        """Sıra numarasına göre tur bilgisini getir"""
        if 0 < number <= len(self.last_shown_tours):
            return self.last_shown_tours[number-1]
        return None

    def clear_temp_tours(self) -> None:
        """Geçici tur listesini temizle"""
        self.temp_tours = []

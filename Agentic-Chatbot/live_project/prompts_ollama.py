
from connection_sql import KeyManager
key_manager = KeyManager()
prompt_instructions = key_manager.load_prompt()

# =========================================================
# NİYET ANALİZİ İÇİN PROMPT
# =========================================================

main_instructions = prompt_instructions ["main_instructions"]

# =========================================================
# SORUDAKİ alanları, payload ve filtering search için gerekli bilgileri çıkaran agent için promptu
# =========================================================
fields_instructions = prompt_instructions ["fields_instructions"]

# =========================================================
# TURUN (tour_info, tour_program_detail) kısmını da kullanmak için çekiyoruz.
# =========================================================
qdrant__id_instructions          = prompt_instructions ["qdrant__id_instructions"]
qdrant_response_instructions     = prompt_instructions ["qdrant_response_instructions"]


# =========================================================
# SMART MEMORY — Özet agent'ı promptu
# =========================================================
summary_instructions = prompt_instructions ["summary_instructions"]

# =========================================================
# TOUR ID RESOLUTION — Fuzzy tur referansı çözümleme
# =========================================================
tour_id_resolve_instructions= prompt_instructions ["tour_id_resolve_instructions"]
output_instructions = prompt_instructions ["output_instructions"]

# =========================================================
# RESERVATION INFO EXTRACTION — Adım adım rezervasyon bilgisi çıkarma
# =========================================================
reservation_extract_instructions = prompt_instructions ["reservation_extract_instructions"]

















from response_structure import (
    ResponseData, 
    SocialRequest,
    SocialResponse,
    TourListItem,
    TourInfoView,
    TourPayload,
    QuestionFieldResult,
)

from agentic_service import (
    main_router_agent,
    question_field_agent,
    tour_information_agent,
    response_agent,
    summary_agent,
    tour_id_resolver_agent,
    qdrant_response_agent,
)
from token_price import Usage, count_tokens
import json, logging, re, asyncio

from connection_qdrant import QdrantDatabase, qdrant_host, qdrant_api_key
client = QdrantDatabase().get_client()
from setting import (
    QDRANT_COLLECTION,
    DENSE_MODEL, SPARSE_MODEL, RERANKER_MODEL
)
from filter_search import FILTER_SEARCH_FIELDS, build_advanced_filter, build_raw_filter, resolve_payload_fields
import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from connection_sql import KeyManager
key_manager = KeyManager()
api_list = key_manager.load_api_key()
model_llm = api_list.get('model_ollama', '')
mng_api_url = api_list.get('mng_api_url', '')

# =========================================================
# region CONCURRENCY & THRESHOLDS
# =========================================================
# vLLM max 7 eşzamanlı konuşma destekler — Semaphore ile kontrol
MAX_CONCURRENT = 7
# previousMessages token sayısı bu eşiği aşarsa → özetleme yapılır
TOKEN_THRESHOLD = 10_000
# Özetleme sonrası korunacak son mesaj sayısı
RECENT_MESSAGES_KEEP = 6

_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# region PAYLOAD → TOON DÖNÜŞÜM
# =========================================================

def flatten_tour_payload(item: dict) -> dict:
    """
    Qdrant'tan gelen nested payload'u düz (flat) ve okunabilir formata çevirir.
    Tüm alanlar handle edilir — nested/array yapılar tek string'e indirgenir.

    Kaynak (Qdrant payload)                 →  Hedef (flat dict)
    ──────────────────────────────────────────────────────────────
    id                                      →  id
    isim                                    →  isim
    geceleme / gecesayisi                   →  sure          "3 Gece 4 Gün"
    turtipi[]                               →  tur_tipi      "yurt dışı"
    konaklama{..., tam_metin}               →  konaklama     "3* - 4* Oteller"
    ulasim + ulasimtipi                     →  ulasim        "THY ile İstanbul - Malta"
    tarihFiyat[{tarih,kisibasi,kur,kalan}]  →  fiyat         ["9 Ekim 2025 (799.0 EUR) Kalan:20", ...]
    kalkissehir                             →  kalkis        "İstanbul"
    kesinkalkis                             →  kesin_kalkis  "Evet" / "Hayır"
    vizesiz                                 →  vize          "Vizesiz" / "Vizeli"
    vizeulke                                →  vize_ulke     "Yunanistan"
    ziyaretedilecekyerler[]                 →  yerler        "Malta, Gozo"
    turkodu                                 →  kod           "AVT323"
    url                                     →  url           "https://..."
    metin                                   →  aciklama      "MNG Turizm güvencesi ile..."
    """
    flat: dict = {}

    # ── id + isim (her zaman) ──
    flat["id"] = item.get("id")
    flat["isim"] = item.get("isim")

    # ── geceleme / gecesayisi → süre ──
    geceleme = item.get("geceleme")
    gecesayisi = item.get("gecesayisi")
    if geceleme:
        flat["sure"] = geceleme
    elif gecesayisi is not None:
        flat["sure"] = f"{gecesayisi} Gece"

    # ── turtipi → tek string ──
    turtipi = item.get("turtipi")
    if turtipi:
        if isinstance(turtipi, list):
            joined = ", ".join(str(t) for t in turtipi if t)
            if joined:
                flat["tur_tipi"] = joined
        else:
            flat["tur_tipi"] = str(turtipi)

    # ── konaklama → tam_metin ──
    konaklama = item.get("konaklama")
    if konaklama and isinstance(konaklama, dict):
        tam_metin = konaklama.get("tam_metin")
        if tam_metin:
            flat["konaklama"] = tam_metin

    # ── ulasim / ulasimtipi → tek alan (detaylı olan tercih edilir) ──
    ulasim = item.get("ulasim")
    ulasimtipi = item.get("ulasimtipi")
    if ulasim:
        flat["ulasim"] = ulasim
    elif ulasimtipi:
        flat["ulasim"] = ulasimtipi

    # ── tarihFiyat → okunabilir string listesi ──
    tarih_fiyat = item.get("tarihFiyat")
    if tarih_fiyat and isinstance(tarih_fiyat, list):
        fiyatlar = []
        for tf in tarih_fiyat:
            parts = []
            tarih = tf.get("tarih") or {}
            gun = tarih.get("gun", "")
            ay = tarih.get("ay", "")
            yil = tarih.get("yil", "")
            tarih_str = f"{gun} {ay} {yil}".strip()
            if tarih_str:
                parts.append(tarih_str)

            kisibasi = tf.get("kisibasi")
            kur = tf.get("kur", "")
            if kisibasi is not None:
                parts.append(f"({kisibasi} {kur})")

            kalan = tf.get("kalan")
            if kalan is not None:
                parts.append(f"Kalan:{kalan}")

            if parts:
                fiyatlar.append(" ".join(parts))
        if fiyatlar:
            flat["fiyat"] = fiyatlar

    # ── kalkissehir ──
    kalkissehir = item.get("kalkissehir")
    if kalkissehir:
        flat["kalkis"] = kalkissehir

    # ── kesinkalkis → kısa format ──
    kesinkalkis = item.get("kesinkalkis")
    if kesinkalkis:
        ks = str(kesinkalkis).lower()
        if "kesin kalkışlıdır" in ks or kesinkalkis == "1":
            flat["kesin_kalkis"] = "Evet"
        elif "kesin kalkışlı değildir" in ks or kesinkalkis == "0":
            flat["kesin_kalkis"] = "Hayır"
        else:
            flat["kesin_kalkis"] = kesinkalkis

    # ── vizesiz → kısa format ──
    vizesiz = item.get("vizesiz")
    if vizesiz:
        vz = str(vizesiz).lower()
        if "vizesizdir" in vz or vizesiz == "0":
            flat["vize"] = "Vizesiz"
        elif "vizelidir" in vz or vizesiz == "1":
            flat["vize"] = "Vizeli"
        else:
            flat["vize"] = vizesiz

    # ── vizeulke ──
    vizeulke = item.get("vizeulke")
    if vizeulke:
        flat["vize_ulke"] = vizeulke

    # ── ziyaretedilecekyerler → virgüllü string ──
    yerler = item.get("ziyaretedilecekyerler")
    if yerler:
        if isinstance(yerler, list):
            joined = ", ".join(str(y) for y in yerler if y)
            if joined:
                flat["yerler"] = joined
        else:
            flat["yerler"] = str(yerler)


    # ── turkodu ──
    turkodu = item.get("turkodu")
    if turkodu:
        flat["kod"] = turkodu

    # ── url ──
    url = item.get("url")
    if url:
        flat["url"] = str(url)

    # ── metin (açıklama) → tek satır ──
    metin = item.get("metin")
    if metin:
        flat["aciklama"] = " ".join(str(metin).split())

    return flat


# TOON çıktısında alan sırası ve etiketleri
_TOON_FIELD_ORDER = [
    ("sure",         "Süre"),
    ("tur_tipi",     "Tur Tipi"),
    ("konaklama",    "Konaklama"),
    ("ulasim",       "Ulaşım"),
    ("fiyat",        "Fiyat"),
    ("kalkis",       "Kalkış"),
    ("kesin_kalkis", "Kesin Kalkış"),
    ("vize",         "Vize"),
    ("vize_ulke",    "Vize Ülke"),
    ("yerler",       "Güzergah Yerleri"),
    ("kod",          "Tur Kodu"),
    ("url",          "URL"),
    ("aciklama",     "Açıklama"),
]


def to_toon(items: list) -> str:
    """
    Tur listesini TOON (Token-Optimized Notation) formatına çevirir.
    JSON'a kıyasla ~%60-70 daha az token kullanır.

    Çıktı örneği (fiyatsız):
    [#171] Malta Turu THY ile 3 Gece | Süre: 3 Gece 4 Gün | Tür: yurt dışı | Konaklama: 3*-4* Oteller | Ulaşım: THY ile İstanbul-Malta | Vize: Vizesiz | Yerler: Malta | Kod: AVT323

    Çıktı örneği (fiyatlı):
    [#171] Malta Turu THY ile 3 Gece | Konaklama: 3*-4* Oteller | Ulaşım: THY
      📊 Fiyat Bilgileri:
        - 9 Ekim 2025 (799.0 EUR) Kalan:20
        - 15 Kasım 2025 (899.0 EUR) Kalan:10
    """
    lines = []
    for item in items:
        parts = [f"[#{item.get('id', '?')}] {item.get('isim', '')}"]
        fiyat_entries: list[str] = []

        # Sabit sırayla tüm alanlar
        for key, label in _TOON_FIELD_ORDER:
            val = item.get(key)
            if val is None:
                continue
            # Fiyat: ayrı satırlarda tablo-uyumlu format
            if key == "fiyat" and isinstance(val, list):
                fiyat_entries = [str(v) for v in val]
            elif isinstance(val, list):
                parts.append(f"{label}: {', '.join(str(v) for v in val)}")
            else:
                parts.append(f"{label}: {val}")

        main_line = " | ".join(parts)
        if fiyat_entries:
            main_line += "\n  📊 Fiyat Bilgileri:"
            for entry in fiyat_entries:
                main_line += f"\n    - {entry}"
        lines.append(main_line)

    return "\n".join(lines)

# endregion
# =========================================================


# =========================================================
# region MORE TOURS — Geçmiş ID Çıkarma
# =========================================================
_TOON_ID_PATTERN = re.compile(r'\[#(\d+)\]')


def extract_shown_tour_ids(chat_history: list[dict]) -> list[int]:
    """
    Chat history'deki TOON çıktılarından daha önce gösterilmiş
    tur ID'lerini çıkarır.  [#677] → 677

    Sadece assistant mesajlarına bakar (kullanıcı mesajlarında
    ID olması durumunu hariç tutar).
    """
    ids: set[int] = set()
    for msg in chat_history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for m in _TOON_ID_PATTERN.findall(content):
            ids.add(int(m))
    return sorted(ids)


def find_original_tour_query(chat_history: list[dict]) -> str | None:
    """
    more_tours: TOON yanıt zincirini geriye doğru takip ederek
    orijinal arama sorgusunu bulur.

    Zincir mantığı:
      user: "yurt içi turlar"      ← BU DÖNER (orijinal sorgu)
      assistant: [#677] ...         (TOON yanıt)
      user: "başka var mı"
      assistant: [#171] ...         (TOON yanıt)
      user: "daha var mı acaba"     ← lastMessage (more_tours)

    Algoritma:
    1. Sondan geriye doğru ilk TOON assistant mesajını bul
    2. Hemen öncesindeki user mesajını al
    3. Eğer o user mesajı da bir TOON'dan sonra gelmişse (yani
       kendisi de "more" tarzı), daha geriye git
    4. Önünde TOON olmayan ilk user mesajı = orijinal sorgu
    """
    idx = len(chat_history) - 1

    while idx >= 0:
        # TOON olmayan mesajları atla
        if chat_history[idx].get("role") != "assistant" or \
           not _TOON_ID_PATTERN.search(chat_history[idx].get("content", "")):
            idx -= 1
            continue

        # TOON bulundu (index `idx`). Hemen önceki user mesajını bul.
        user_idx = None
        for j in range(idx - 1, -1, -1):
            if chat_history[j].get("role") == "user":
                user_idx = j
                break

        if user_idx is None:
            return None  # TOON'dan önce user mesajı yok

        # Bu user mesajının hemen öncesinde de TOON var mı?
        # Varsa → bu user "more" zincirinin parçası, daha geriye git
        has_preceding_toon = False
        for k in range(user_idx - 1, -1, -1):
            if chat_history[k].get("role") == "assistant":
                if _TOON_ID_PATTERN.search(chat_history[k].get("content", "")):
                    has_preceding_toon = True
                    idx = k      # Önceki TOON'a atla
                break            # İlk assistant'ta dur

        if not has_preceding_toon:
            return chat_history[user_idx]["content"]
        # has_preceding_toon → while döngüsü idx=k ile devam eder

    return None


# endregion
# =========================================================


# =========================================================
# region SMART MEMORY MANAGEMENT
# =========================================================

async def smart_memory_management(
    chat_history: list[dict],
    total_usage: Usage,
    threshold: int = TOKEN_THRESHOLD,
) -> list[dict]:
    """
    previousMessages token sayısını hesaplar.
    Eşik aşılırsa eski mesajları LLM ile özetler, son N mesajı korur.

    Dönen yapı:
        [
          {"role": "system", "content": "Önceki konuşma özeti: ..."},
          ...recent_messages (son RECENT_MESSAGES_KEEP mesaj)
        ]

    Sayısal değerler (tur ID'leri, fiyatlar, tarihler) özette korunur.
    """
    if not chat_history:
        return chat_history

    # ── Token hesapla ──
    total_text = "\n".join(m.get("content", "") for m in chat_history)
    total_tokens = count_tokens(total_text)

    logger.info(
        "📊 Chat history: %d mesaj, ~%d token (threshold: %d)",
        len(chat_history), total_tokens, threshold,
    )

    if total_tokens <= threshold:
        return chat_history  # Özetlemeye gerek yok

    # Çok az mesaj varsa özetleme anlamsız
    if len(chat_history) <= RECENT_MESSAGES_KEEP:
        return chat_history

    # ── Böl: eski + son mesajlar ──
    old_messages = chat_history[:-RECENT_MESSAGES_KEEP]
    recent_messages = chat_history[-RECENT_MESSAGES_KEEP:]

    logger.info(
        "📝 Özetleme: %d eski mesaj özetlenecek, %d mesaj korunacak",
        len(old_messages), len(recent_messages),
    )

    # ── Eski mesajları metin olarak hazırla ──
    old_text_parts: list[str] = []
    for m in old_messages:
        role_label = "Kullanıcı" if m["role"] == "user" else "Asistan"
        old_text_parts.append(f"{role_label}: {m['content']}")
    old_text = "\n".join(old_text_parts)

    summary_input = (
        "Aşağıdaki konuşma geçmişini kısa ve öz şekilde özetle. "
        "Sayısal değerleri (tur ID'leri [#ID], fiyatlar, tarihler, kontenjanlar) "
        "ASLA değiştirme, olduğu gibi koru:\n\n"
        f"{old_text}"
    )

    # ── LLM ile özetle ──
    summary_result = await summary_agent.run(summary_input, [])

    if isinstance(summary_result, dict):
        usage = summary_result.get("usage")
        if usage:
            total_usage.add(usage)
        summary_text = summary_result.get("text", "")
    else:
        summary_text = str(summary_result)

    logger.info("✅ Özet üretildi (%d karakter)", len(summary_text))

    # ── Yeni geçmiş: [özet_system] + son mesajlar ──
    new_history = [
        {"role": "system", "content": f"Önceki konuşma özeti: {summary_text}"}
    ] + recent_messages

    new_tokens = count_tokens(
        "\n".join(m.get("content", "") for m in new_history)
    )
    logger.info(
        "📊 Yeni token sayısı: %d (önceki: %d, tasarruf: %%%d)",
        new_tokens, total_tokens,
        int((1 - new_tokens / max(total_tokens, 1)) * 100),
    )

    return new_history


# endregion
# =========================================================


# =========================================================
# region FUZZY TOUR ID RESOLUTION
# =========================================================

_TOON_TOUR_PATTERN = re.compile(r'\[#(\d+)\]\s*(.+?)(?:\s*\||\s*$)', re.MULTILINE)

# Fallback: "📌 Tur Adı (THY ile 3 gece)" veya "📌 Tur Adı — detay" gibi emoji başlıklı satırlar
_EMOJI_TOUR_PATTERN = re.compile(
    r'📌\s*(?:\[#(\d+)\])?\s*(.+?)(?:\s*$)',
    re.MULTILINE,
)


def extract_shown_tours_with_context(chat_history: list[dict]) -> list[dict]:
    """
    Chat history'deki TOON çıktılarından tur ID + isim çiftlerini çıkarır.
    Son gösterilen turlar ÖNCE gelecek şekilde sıralanır (reverse chronological).

    Önce [#ID] TOON formatı aranır; bulunamazsa 📌 ile başlayan satırlar
    da parse edilerek tur listesi çıkarılır (ID'siz turlar position-based
    olarak işlenir).

    Dönüş: [{"id": 677, "name": "Malta Turu", "position": 1}, ...]
    """
    tours: list[dict] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for msg in reversed(chat_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")

        # ── 1) TOON formatı: [#ID] Tur Adı ──
        msg_tours: list[dict] = []
        for match in _TOON_TOUR_PATTERN.finditer(content):
            tour_id = int(match.group(1))
            tour_name = match.group(2).strip()
            if tour_id not in seen_ids:
                msg_tours.append({"id": tour_id, "name": tour_name})
                seen_ids.add(tour_id)

        # ── 2) Fallback: 📌 satırları (ID varsa kullan, yoksa 0) ──
        if not msg_tours:
            for match in _EMOJI_TOUR_PATTERN.finditer(content):
                raw_id = match.group(1)
                tour_name = match.group(2).strip().rstrip(" .")
                if not tour_name:
                    continue
                # İsim temizle: parantez içi detayları koru ama sondaki boşlukları sil
                tour_id = int(raw_id) if raw_id else 0
                name_key = tour_name.lower()
                if tour_id and tour_id not in seen_ids:
                    msg_tours.append({"id": tour_id, "name": tour_name})
                    seen_ids.add(tour_id)
                    seen_names.add(name_key)
                elif not tour_id and name_key not in seen_names:
                    msg_tours.append({"id": 0, "name": tour_name})
                    seen_names.add(name_key)

        # Mesaj içi sırayı koru (pozisyon ata)
        for idx, t in enumerate(msg_tours, 1):
            t["position"] = idx
        tours.extend(msg_tours)

    return tours


def _find_last_mentioned_tour(chat_history: list[dict]) -> dict | None:
    """
    En son konuşulan turu bulur.
    
    Strateji:
    1. Chat history'yi sondan başa tara
    2. En son assistant mesajında tek bir tur hakkında konuşuluyorsa → o tur
    3. En son user mesajında bir tur referansı varsa → o tur
    4. Yoksa en son assistant mesajındaki ilk tur
    """
    # Son birkaç mesajı kontrol et (en son konuşulan bağlamı bul)
    for msg in reversed(chat_history):
        content = msg.get("content", "")
        role = msg.get("role", "")
        
        # Assistant mesajında tek bir [#ID] varsa, o tur hakkında konuşuluyor demektir
        if role == "assistant":
            matches = list(_TOON_TOUR_PATTERN.finditer(content))
            if len(matches) == 1:
                # Tek tur hakkında konuşuluyor
                tour_id = int(matches[0].group(1))
                tour_name = matches[0].group(2).strip()
                return {"tour_id": tour_id, "tour_name": tour_name}
            elif len(matches) > 1:
                # Birden fazla tur var, ilkini al (en son bahsedilen)
                tour_id = int(matches[0].group(1))
                tour_name = matches[0].group(2).strip()
                return {"tour_id": tour_id, "tour_name": tour_name}
    
    return None


# ── Sıra sayısı regex: "4. tur", "4.tur", "dördüncü tur" vb. ──
_ORDINAL_PATTERN = re.compile(r'(\d+)\s*\.\s*tur', re.IGNORECASE)

_ORDINAL_WORDS: dict[str, int] = {
    "birinci": 1, "ilk": 1, "1.": 1,
    "ikinci": 2, "2.": 2,
    "üçüncü": 3, "3.": 3,
    "dördüncü": 4, "4.": 4,
    "beşinci": 5, "5.": 5,
    "altıncı": 6, "6.": 6,
    "yedinci": 7, "7.": 7,
    "sekizinci": 8, "8.": 8,
    "dokuzuncu": 9, "9.": 9,
    "onuncu": 10, "10.": 10,
    "son": -1, "sonuncu": -1, "en alttaki": -1,
}


def _extract_ordinal(text: str) -> int | None:
    """Sorudan sıra numarası çıkar: '4. tur' → 4, 'dördüncü tur' → 4."""
    m = _ORDINAL_PATTERN.search(text)
    if m:
        return int(m.group(1))
    lower = text.lower()
    for word, idx in _ORDINAL_WORDS.items():
        if word in lower:
            return idx
    return None


async def resolve_tour_id(
    soru: str,
    chat_history: list[dict],
    total_usage: Usage,
) -> dict | None:
    """
    Kullanıcının referans verdiği turu belirler.
    "ilk tur", "şu kıbrıs turu", "az önceki", "#677", "4. tur" gibi ifadeleri çözer.

    Returns: {"tour_id": int, "tour_name": str} veya None
    """
    # ── Direkt ID referansı kontrol et: "#677", "677 numaralı" ──
    direct_id_match = re.search(r'#(\d{2,})', soru)
    if direct_id_match:
        tour_id = int(direct_id_match.group(1))
        logger.info("🎯 Direkt tour ID bulundu: %d", tour_id)
        return {"tour_id": tour_id, "tour_name": ""}

    # ── Chat geçmişindeki turları çıkar ──
    shown_tours = extract_shown_tours_with_context(chat_history)
    if not shown_tours:
        logger.warning("⚠️ Geçmişte gösterilmiş tur bulunamadı")
        return None

    logger.info(
        "🔍 Geçmişten bulunan turlar (%d adet): %s",
        len(shown_tours),
        [(t["id"], t["name"][:40]) for t in shown_tours],
    )

    # ── "bu tur", "o tur", "bahsettiğim tur" gibi belirsiz referanslar ──
    # En son konuşulan turu (chat_history'deki en son assistant mesajında
    # bahsedilen ilk turu) direkt döndür, LLM'e sormadan.
    soru_lower = soru.lower()
    _LAST_TOUR_KEYWORDS = [
        "bu tur", "o tur", "şu tur", "bu turun", "o turun", "şu turun",
        "az önceki", "bahsettiğim", "konuştuğumuz", "gösterdiğin",
        "son gösterdiğin", "detayları", "detaylarını", "programı",
    ]
    if any(kw in soru_lower for kw in _LAST_TOUR_KEYWORDS):
        # En son konuşulan turu bul (en son assistant mesajında bahsedilen ilk tur)
        last_mentioned = _find_last_mentioned_tour(chat_history)
        if last_mentioned:
            logger.info(
                "🎯 'Bu tur' referansı → En son konuşulan tur: ID=%d (%s)",
                last_mentioned["tour_id"], last_mentioned["tour_name"],
            )
            return last_mentioned

    # ── Sıra numarası kontrolü: "4. tur" → position 4 ──
    ordinal = _extract_ordinal(soru)
    if ordinal is not None:
        if ordinal == -1:
            # "son tur"
            target = shown_tours[-1] if shown_tours else None
        elif 1 <= ordinal <= len(shown_tours):
            target = shown_tours[ordinal - 1]
        else:
            target = None

        if target:
            logger.info(
                "🎯 Sıra referansı ile eşleşti: pos=%s → id=%d name=%s",
                ordinal, target["id"], target["name"],
            )
            return {"tour_id": target["id"], "tour_name": target["name"]}

    # ── Turları LLM'e gönder, fuzzy eşleştirme yap ──
    tours_context = "\n".join(
        f"{idx+1}. [#{t['id']}] {t['name']}" if t['id'] else f"{idx+1}. {t['name']}"
        for idx, t in enumerate(shown_tours)
    )

    resolve_input = (
        f"Gösterilmiş turlar (son gösterilenler önce):\n"
        f"{tours_context}\n\n"
        f"Kullanıcı sorusu: {soru}\n\n"
        f"Kullanıcının hangi tura referans verdiğini belirle."
    )

    result = await tour_id_resolver_agent.run(resolve_input, chat_history[-4:])

    if isinstance(result, dict):
        usage = result.get("usage")
        if usage:
            total_usage.add(usage)
        tour_id = result.get("tour_id")
        if tour_id is not None:
            try:
                return {
                    "tour_id": int(tour_id),
                    "tour_name": result.get("tour_name", ""),
                }
            except (ValueError, TypeError):
                pass

    logger.warning("⚠️ Tur ID çözümlenemedi")
    return None


# endregion
# =========================================================


# =========================================================
# region MNG API — Tur Programı Detayı
# =========================================================

async def call_mng_tour_program(tour_id: int) -> str | None:
    """
    MNG Turizm servisinden tur programı detaylarını çeker.
    POST https://www.example.com/index.php
    Form: s=aisorgu&action=tour-program&data[id]=TOUR_ID
    Yanıt aynen kullanıcıya gösterilir (raw text).
    """
    MNG_URL = mng_api_url or "https://www.example.com/index.php"
    MNG_AUTH = "Basic bW5nYWl1c2VyOm1uZ2FpcGFzcw=="

    try:
        form_data = f"s=aisorgu&action=tour-program&data%5Bid%5D={tour_id}"

        async with httpx.AsyncClient(timeout=30, verify=False) as http_client:
            resp = await http_client.post(
                MNG_URL,
                content=form_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": MNG_AUTH,
                },
            )
            if resp.status_code == 200:
                logger.info("✅ MNG API yanıt aldı (tour_id=%d)", tour_id)
                return resp.text
            else:
                logger.error(
                    "❌ MNG API hatası: %d — %s", resp.status_code, resp.text
                )
                return None
    except httpx.RequestError as e:
        logger.error("❌ MNG API bağlantı hatası: %s", e)
        return None


# endregion
# =========================================================


# =========================================================
# region STYLE APPLICATOR (aiSettings)
# =========================================================

def build_styled_input(
    base_input: str,
    ai_prompt_content: str | None = None,
) -> str:
    """
    aiSettings'den gelen stil talimatını input'a ekler.
    SADECE ton/stil etkiler — mantık/filtreleme/veri doğruluğu değişmez.
    """
    if ai_prompt_content and ai_prompt_content not in ("N/A", "", None):
        return (
            f"{base_input}\n\n"
            f"[Yanıt Stili: {ai_prompt_content}. "
            f"Bu talimat SADECE yanıtın tonunu ve stilini etkiler, "
            f"veri doğruluğunu veya filtreleme mantığını değiştirmez.]"
        )
    return base_input


# endregion
# =========================================================



def finalize_with_payload(fields: list[str]) -> list[str]:
    """
    with_payload listesini Qdrant'a göndermeden önce son düzeltmeleri yapar:
    1. 'konaklama' → 'konaklama.tam_metin'  (sadece tam_metin çekilir, tüm nesne değil)
    2. ulasimtipi varsa ulasim da eklenir (veya tam tersi) — ikisi birlikte gelir
    """
    result: list[str] = []
    for f in fields:
        if f == "konaklama":
            result.append("konaklama.tam_metin")
        else:
            result.append(f)

    # ulasimtipi ↔ ulasim birliktelik kuralı
    has_ulasimtipi = "ulasimtipi" in result
    has_ulasim = "ulasim" in result
    if has_ulasimtipi and not has_ulasim:
        result.append("ulasim")
    elif has_ulasim and not has_ulasimtipi:
        result.append("ulasimtipi")

    return list(dict.fromkeys(result))  # deduplicate


def get_user_addressing(firstName: str, gender: str) -> str:
    """Kullanıcı adı ve cinsiyetine göre uygun hitap şeklini döndür"""
    if gender.upper() == "M":
        return f"{firstName} Bey"
    elif gender.upper() == "F":
        return f"{firstName} Hanım"
    else:
        return firstName
# program çözümleyici

def format_tour_program(cleaned):
    result = []

    days = cleaned.get("data", {})

    for key in sorted(days, key=lambda x: int(x)):
        day = days[key]

        gun = day.get("gun", "")
        baslik = day.get("baslik", "")
        program = day.get("program", "").replace("\\r\\n", "\n")
        varis = day.get("cruisevaris", "")
        kalkis = day.get("cruisekalkis", "")

        # ✅ başlık (senin beğendiğin)
        block = f"\n📍 {gun}. Gün  |  {baslik}\n"

        if varis:
            block += f"🚢 Varış: {varis}\n"
        if kalkis:
            block += f"🛫 Kalkış: {kalkis}\n"

        block += f"\n{program}\n"
        block += "─" * 50

        result.append(block)

    return "\n".join(result)

def clean_data(obj):
    """JSON parse sonrası string temizliği (encoding dönüşümü YAPMAZ)."""
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(i) for i in obj]
    elif isinstance(obj, str):
        # Sadece literal escape sequence'leri temizle
        return obj.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    else:
        return obj

# çocuk yaşı hesaplama ve turun fiyat hesaplaması için gerekli fonksiyon,
def tatil_ucreti_hesapla(yetiskin_sayisi, yetiskin_fiyat, cocuk_yaslari):
    toplam_ucret = yetiskin_sayisi * yetiskin_fiyat
    
    for yas in cocuk_yaslari:
        if yas <= 6:
            # 0-6 yaş ücretsiz
            cocuk_ucreti = 0
        elif 7 <= yas <= 12:
            # 7-12 yaş %50 indirimli
            cocuk_ucreti = yetiskin_fiyat * 0.5
        else:
            # 13 yaş ve üzeri yetişkin kabul edilir
            cocuk_ucreti = yetiskin_fiyat
            
        toplam_ucret += cocuk_ucreti
        
    return toplam_ucret



async def handle_request(request: SocialRequest) -> SocialResponse:
    """
    Ana giriş noktası — Semaphore ile eşzamanlılık kontrolü sağlar.
    chatId bazında izolasyon: her istek bağımsız işlenir.
    """
    async with _request_semaphore:
        return await _process_request(request)


async def _process_request(request: SocialRequest) -> SocialResponse:
    """
    Asıl iş mantığı. Semaphore tarafından kontrol edilen iç fonksiyon.

    Pipeline:
      1. Kullanıcı verileri hazırla
      2. Chat history oluştur
      3. Smart memory management (token > threshold → özetle)
      4. Intent agent (sınıflandırma)
      5. Route → uygun handler
      6. aiSettings ile stil uygula
      7. Structured response döndür
    """
    try:
        logger.info(
            "\n=== YENİ MESAJ İŞLEME BAŞLADI (chatId: %s) ===",
            request.chatId,
        )

        contract_type = request.agent_parametre
        logger.info(f"Agent parametresi (contract type): {contract_type}")

        if contract_type != "free":
            return SocialResponse(
                success=False,
                errorMessage=f"Bilinmeyen Sözleşme Türü!!! Paket: {contract_type}",
            )

        # =============================================================
        # 📌 KULLANICI VERİLERİ HAZIRLAMA
        # =============================================================
        user_addressing = get_user_addressing(
            request.customer.firstName,
            request.customer.gender,
        )

        soru = request.lastMessage

        aisetting = (
            request.aiSettings.prompts[0]
            if request.aiSettings.prompts
            else None
        )
        ai_prompt_type = aisetting.type if aisetting else "N/A"
        ai_prompt_content = aisetting.prompt if aisetting else "N/A"

        # -------------------------------------------------------------
        # 🔹 Chat History → OpenAI messages formatı
        # -------------------------------------------------------------
        chat_history: list[dict] = [
            {"role": msg.role, "content": msg.content}
            for msg in request.previousMessages
        ]
        logger.info(f"Chat history mesaj sayısı: {len(chat_history)}")

        # =============================================================
        # 📌 TOPLAM USAGE
        # =============================================================
        total_usage = Usage()

        # =============================================================
        # 📌 SMART MEMORY MANAGEMENT
        # Token > TOKEN_THRESHOLD ise eski mesajları özetle, son N mesajı koru.
        # Sayısal değerler ve tur ID'leri özette korunur.
        # =============================================================
        chat_history = await smart_memory_management(
            chat_history, total_usage
        )

        # -------------------------------------------------------------
        # 🔹 Agent'lara gönderilecek input metinleri
        # -------------------------------------------------------------
        router_input = (
            f"Kullanıcı: {user_addressing}\n"
            f"Soru: {soru}"
        )

        response_input = build_styled_input(
            f"Kullanıcı: {user_addressing}\n"
            f"Soru: {soru}",
            ai_prompt_content,
        )

        # =============================================================
        # 1️⃣  INTENT AGENT  (chat_history ile bağlam kurarak sınıflandır)
        # Agent prompt'lar aracılığıyla intent tespiti yapılır.
        # Bozuk Türkçe, eksik sesli harfler, yazım hataları
        # agent tarafından doğal olarak anlaşılır.
        # =============================================================
        intent_result = await main_router_agent.run(router_input, chat_history)

        if isinstance(intent_result, dict):
            usage = intent_result.get("usage")
            if usage:
                total_usage.add(usage)

        raw_label = intent_result.get("types", ["unknown"])
        VALID_LABELS = [
            "general_conversation",
            "chat_summary",
            "out_of_scope",
            "tour_list",
            "more_tours",
            "tour_info",
            "tour_program_detail",
        ]
        labels = [
            label for label in raw_label if label in VALID_LABELS
        ] or ["unknown"]
        logger.info(f"🏷️  Tespit edilen etiketler: {labels}")

        # =============================================================
        # 2️⃣  ROUTING
        # =============================================================
        response_content = ""

        # ─────────────────────────────────────────────────────────────
        # TOUR LIST / MORE TOURS
        # Doğrudan çalıştırılır, onay beklenmez.
        # Agent prompt'lar yazım hataları / eksik harfleri anlar.
        # ─────────────────────────────────────────────────────────────
        if "tour_list" in labels or "more_tours" in labels:

            # ── more_tours → Orijinal sorguyu chat_history'den bul ──
            field_query = soru
            if "more_tours" in labels:
                original = find_original_tour_query(chat_history)
                if original:
                    field_query = original
                    logger.info(
                        "🔄 more_tours: Orijinal sorgu kullanılıyor → '%s'",
                        original,
                    )
                else:
                    logger.warning(
                        "⚠️ more_tours: Orijinal sorgu bulunamadı, "
                        "mevcut soru kullanılacak"
                    )

            # ── question_field_agent: Alan & filtre çıkarımı ──
            qdrant_result = await question_field_agent.run(
                field_query, chat_history
            )
            logger.info(f"QuestionFieldAgent RAW sonucu: {qdrant_result}")

            # ── Pydantic ile alan adı doğrulama + fuzzy düzeltme ──
            validated, corrections = QuestionFieldResult.validate_agent_output(
                qdrant_result
            )
            if corrections:
                logger.warning("🔧 Alan düzeltmeleri: %s", corrections)

            label_field = validated.label_field
            kume_list = validated.kume_list
            agent_payload_fields = validated.payload_field

            logger.info(f"kume_list                 : {kume_list}")
            logger.info(f"label_field               : {label_field}")
            logger.info(f"agent_payload_fields      : {agent_payload_fields}")

            # ---------------------------------------------------------
            # 🔹 filter_dict oluştur
            # ---------------------------------------------------------
            filter_dict: dict = {}
            for group_name in ["must", "should", "must_not"]:
                group_fields = kume_list.get(group_name, [])
                group_data = {
                    f: label_field[f]
                    for f in group_fields
                    if f in label_field
                    and f in FILTER_SEARCH_FIELDS
                    and label_field[f] is not None
                    and label_field[f] != ""
                    and label_field[f] != []
                }
                if group_data:
                    filter_dict[group_name] = group_data

            # ---------------------------------------------------------
            # 🔹 MORE_TOURS → Daha önce gösterilen tur ID'lerini dışla
            # ---------------------------------------------------------
            if "more_tours" in labels:
                shown_ids = extract_shown_tour_ids(chat_history)
                if shown_ids:
                    must_not_data = filter_dict.get("must_not", {})
                    must_not_data["id"] = shown_ids
                    filter_dict["must_not"] = must_not_data
                    logger.info(
                        "🚫 more_tours: Dışlanan ID'ler (%d adet): %s",
                        len(shown_ids),
                        shown_ids,
                    )
                else:
                    logger.warning(
                        "⚠️ more_tours: Geçmişte gösterilmiş tur ID'si bulunamadı"
                    )

            logger.info(
                "📌 FILTER_DICT: %s",
                json.dumps(
                    filter_dict, indent=2, ensure_ascii=False, default=str
                ),
            )

            # ---------------------------------------------------------
            # 🔹 SEARCH: filtre dolu → FILTERED, boş → EMBEDDING
            # ---------------------------------------------------------
            if filter_dict:
                logger.info("🔹 FILTERED SEARCH çağrısı yapılacak")

                # ── with_payload hesaplama ──
                DEFAULT_FIELDS = ["id", "isim"]
                extra_from_label = [
                    k
                    for k, v in label_field.items()
                    if v is None or v == "" or v == []
                ]
                all_extra = list(
                    dict.fromkeys(agent_payload_fields + extra_from_label)
                )
                extra_resolved = resolve_payload_fields(all_extra)
                with_payload = list(
                    dict.fromkeys(DEFAULT_FIELDS + extra_resolved)
                )
                with_payload = finalize_with_payload(with_payload)

                logger.info("📦 with_payload: %s", with_payload)

                # ── Qdrant HTTP POST ──
                raw_qdrant_filter = build_raw_filter(filter_dict)
                scroll_body: dict = {
                    "filter": raw_qdrant_filter,
                    "limit": 10,
                    "with_payload": with_payload,
                    "with_vector": False,
                }

                logger.info(
                    "🔍 Qdrant scroll body:\n%s",
                    json.dumps(scroll_body, indent=2, ensure_ascii=False),
                )

                async with httpx.AsyncClient(verify=False) as http_qdrant:
                    qdrant_resp = await http_qdrant.post(
                        f"{qdrant_host}/collections/{QDRANT_COLLECTION}"
                        f"/points/scroll",
                        json=scroll_body,
                        headers={
                            "api-key": qdrant_api_key,
                            "Content-Type": "application/json",
                        },
                        timeout=60,
                    )
                    qdrant_data = qdrant_resp.json()

                raw_points_data = qdrant_data.get("result", {}).get(
                    "points", []
                )
                logger.info(
                    "Qdrant'dan dönen nokta sayısı: %d", len(raw_points_data)
                )

                # ── Pydantic → Flatten → TOON ──
                results_flat: list[dict] = []
                for pt in raw_points_data:
                    payload = pt.get("payload", {})
                    if payload:
                        try:
                            item = TourListItem.model_validate(payload)
                            dumped = item.model_dump(exclude_none=True)
                            flat = flatten_tour_payload(dumped)
                            results_flat.append(flat)
                        except Exception as e:
                            logger.warning(
                                "Payload işlenirken hata: %s — %s", e, payload
                            )

                toon_output = to_toon(results_flat)
                logger.info(
                    "📋 TOON çıktı (%d tur):\n%s",
                    len(results_flat),
                    toon_output,
                )

                # ── TOON verilerini doğal dile çevir (qdrant_response_agent) ──
                if results_flat:
                    qdrant_resp_input = build_styled_input(
                        f"Kullanıcı: {user_addressing}\n"
                        f"Qdrant Sonuçları (TOON format):\n{toon_output}\n\n"
                        f"Soru: {field_query}",
                        ai_prompt_content,
                    )
                    resp_result = await qdrant_response_agent.run(
                        qdrant_resp_input, chat_history
                    )
                    if isinstance(resp_result, dict):
                        usage = resp_result.get("usage")
                        if usage:
                            total_usage.add(usage)
                        response_content = resp_result.get("text", toon_output)
                    else:
                        response_content = str(resp_result)
                else:
                    response_content = (
                        f"{user_addressing}, aradığınız kriterlere uygun "
                        f"tur bulunamadı. Farklı kriterler deneyebilir miyiz? 😊"
                    )

            else:
                logger.info(
                    "🔹 EMBEDDING SEARCH yapılacak (filtre alanı bulunamadı)"
                )
                # TODO: embedding search implementasyonu
                response_content = (
                    f"{user_addressing}, aramanızı daha iyi "
                    f"anlamam için birkaç detay verir misiniz? "
                    f"Örneğin gece sayısı, bölge veya tur tipi gibi. 😊"
                )

        # ─────────────────────────────────────────────────────────────
        # TOUR INFO — Belirli bir turun detayları
        # Doğrudan çalıştırılır, onay beklenmez.
        # Agent tur referansını çözer (fuzzy: "ilk tur", "kıbrıs turu")
        # ─────────────────────────────────────────────────────────────
        elif "tour_info" in labels:

            logger.info("🔎 tour_info handler başlatıldı")

            # ── 1. Hangi alanlar isteniyor? (question_field_agent) ──
            field_result = await question_field_agent.run(soru, chat_history)
            if isinstance(field_result, dict):
                usage = field_result.get("usage")
                if usage:
                    total_usage.add(usage)

            # RAW sonucu logla (debug)
            logger.info(
                "🔎 tour_info QuestionFieldAgent RAW sonucu: %s",
                {k: v for k, v in field_result.items() if k != "usage"}
                if isinstance(field_result, dict) else field_result,
            )

            validated, corrections = QuestionFieldResult.validate_agent_output(
                field_result
            )
            if corrections:
                logger.warning("🔧 tour_info alan düzeltmeleri: %s", corrections)

            requested_payload = validated.payload_field
            label_field = validated.label_field

            logger.info(
                "🔎 tour_info validated → label_field: %s | payload_field: %s",
                label_field, requested_payload,
            )

            # ── 2. Tur ID'sini belirle ──
            tour_id_val = label_field.get("id")
            tour_id = None

            if tour_id_val:
                tour_id = int(tour_id_val)
                logger.info("🎯 tour_info: label_field'den ID bulundu: %d", tour_id)
            else:
                # Fuzzy resolution: "ilk tur", "şu kıbrıs turu", "az önceki"
                tour_resolved = await resolve_tour_id(
                    soru, chat_history, total_usage
                )
                if tour_resolved:
                    tour_id = tour_resolved["tour_id"]
                    logger.info(
                        "🎯 tour_info: Fuzzy çözüm → ID=%d (%s)",
                        tour_id,
                        tour_resolved.get("tour_name", ""),
                    )

            if tour_id is None or tour_id == 0:
                if tour_id == 0:
                    logger.warning(
                        "⚠️ Tur ID=0 (geçmişte ID'siz listelenmiş). "
                        "Kullanıcıdan tur numarası isteniyor."
                    )
                response_content = (
                    f"{user_addressing}, hangi turun detaylarını görmek "
                    f"istediğinizi belirtir misiniz? Tur numarası veya "
                    f"adıyla söyleyebilirsiniz. 😊"
                )
            else:
                # ── 3. with_payload: istenilen alanlar + default ──
                DEFAULT_INFO_FIELDS = ["id", "isim"]
                extra_from_label = [
                    k for k, v in label_field.items()
                    if (v is None or v == "" or v == []) and k != "id"
                ]
                all_payload = list(dict.fromkeys(
                    requested_payload + extra_from_label
                ))

                resolved_payload = resolve_payload_fields(all_payload)
                with_payload = list(dict.fromkeys(
                    DEFAULT_INFO_FIELDS + resolved_payload
                ))

                # Eğer hiç ek alan istenmemişse, tüm önemli alanları getir
                if len(with_payload) <= 2:
                    with_payload = [
                        "id", "isim", "geceleme", "gecesayisi",
                        "turtipi", "konaklama.tam_metin", "ulasim", "ulasimtipi",
                        "tarihFiyat", "kalkissehir", "kesinkalkis",
                        "vizesiz", "vizeulke", "ziyaretedilecekyerler",
                        "turkodu", "url",
                    ]

                # konaklama → konaklama.tam_metin, ulasimtipi ↔ ulasim birliktelik
                with_payload = finalize_with_payload(with_payload)

                logger.info(
                    "📦 tour_info with_payload: %s (tour_id=%d)",
                    with_payload, tour_id,
                )

                # ── 4. Qdrant ID-based scroll ──
                id_filter = {"must": [{"key": "id", "match": {"value": tour_id}}]}
                scroll_body = {
                    "filter": id_filter,
                    "limit": 1,
                    "with_payload": with_payload,
                    "with_vector": False,
                }

                async with httpx.AsyncClient(verify=False) as http_qdrant:
                    qdrant_resp = await http_qdrant.post(
                        f"{qdrant_host}/collections/{QDRANT_COLLECTION}"
                        f"/points/scroll",
                        json=scroll_body,
                        headers={
                            "api-key": qdrant_api_key,
                            "Content-Type": "application/json",
                        },
                        timeout=60,
                    )
                    qdrant_data = qdrant_resp.json()

                raw_points = qdrant_data.get("result", {}).get("points", [])

                if raw_points:
                    payload = raw_points[0].get("payload", {})
                    try:
                        item = TourPayload.model_validate(payload)
                        dumped = item.model_dump(exclude_none=True)
                        flat = flatten_tour_payload(dumped)
                        toon_output = to_toon([flat])

                        # ── 5. qdrant_response_agent ile doğal dile çevir ──
                        info_input = build_styled_input(
                            f"Kullanıcı: {user_addressing}\n"
                            f"Qdrant Sonuçları:\n{toon_output}\n\n"
                            f"Soru: {soru}",
                            ai_prompt_content,
                        )
                        resp_result = await qdrant_response_agent.run(
                            info_input, chat_history
                        )
                        if isinstance(resp_result, dict):
                            usage = resp_result.get("usage")
                            if usage:
                                total_usage.add(usage)
                            response_content = resp_result.get("text", toon_output)
                        else:
                            response_content = str(resp_result)

                    except Exception as e:
                        logger.warning("tour_info payload hatası: %s", e)
                        response_content = (
                            f"{user_addressing}, tur bilgilerini getirirken "
                            f"bir sorun oluştu. Tekrar deneyebilir misiniz?"
                        )
                else:
                    response_content = (
                        f"{user_addressing}, {tour_id} numaralı tur "
                        f"bulunamadı. Farklı bir tur belirtir misiniz? 😊"
                    )

            logger.info("✅ tour_info tamamlandı")

        # ─────────────────────────────────────────────────────────────
        # TOUR PROGRAM DETAIL — MNG API üzerinden program detayı
        # Doğrudan çalıştırılır, onay beklenmez.
        # Agent tur referansını çözer → MNG API'ye POST → yanıt aynen gösterilir.
        # ─────────────────────────────────────────────────────────────
        elif "tour_program_detail" in labels:

            logger.info("📋 tour_program_detail handler başlatıldı")

            # ── 1. Tur ID belirle ──
            direct_match = re.search(r'#?(\d{2,})', soru)
            tour_id = None

            if direct_match:
                tour_id = int(direct_match.group(1))
                logger.info("🎯 tour_program: Direkt ID bulundu: %d", tour_id)
            else:
                tour_resolved = await resolve_tour_id(
                    soru, chat_history, total_usage
                )
                if tour_resolved:
                    tour_id = tour_resolved["tour_id"]
                    logger.info(
                        "🎯 tour_program: Fuzzy çözüm → ID=%d (%s)",
                        tour_id,
                        tour_resolved.get("tour_name", ""),
                    )

            if tour_id is None:
                response_content = (
                    f"{user_addressing}, hangi turun programını görmek "
                    f"istediğinizi belirtir misiniz? Tur numarası veya "
                    f"adıyla söyleyebilirsiniz. 😊"
                )
            else:
                logger.info(
                    "📋 MNG API'ye tur programı isteği: tour_id=%d", tour_id
                )

                # ── 2. MNG API'ye POST ──
                mng_responses = await call_mng_tour_program(tour_id)

                # 1. JSON parse et (mng_responses zaten string)
                mng_response = None
                if mng_responses:
                    try:
                        data = json.loads(mng_responses)
                        cleaned = clean_data(data)
                        mng_response = format_tour_program(cleaned)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning("⚠️ MNG yanıtı parse hatası: %s", e)
                        mng_response = mng_responses  # parse başarısızsa raw text

                if mng_response:
                    # MNG yanıtını aynen kullanıcıya göster
                    response_content = mng_response
                    logger.info(
                        "✅ MNG API yanıtı gösterildi (%d karakter)",
                        len(response_content),
                    )
                else:
                    response_content = (
                        f"{user_addressing}, {tour_id} numaralı turun "
                        f"programı şu an alınamadı. Lütfen biraz sonra "
                        f"tekrar deneyiniz. 🙏"
                    )

            logger.info("✅ tour_program_detail tamamlandı")

        # ─────────────────────────────────────────────────────────────
        # CHAT SUMMARY — Geçmiş konuşma özeti
        # ─────────────────────────────────────────────────────────────
        elif "chat_summary" in labels:

            if not chat_history:
                response_content = (
                    f"{user_addressing}, henüz birlikte konuştuğumuz "
                    f"bir konu bulunmuyor. Nasıl yardımcı olabilirim? 😊"
                )
                logger.info("📋 chat_summary: Geçmiş yok, boş yanıt döndü")
            else:
                MAX_MSGS = 20
                recent = chat_history[-MAX_MSGS:]
                summary_lines = []
                for msg in recent:
                    role_label = (
                        "Siz" if msg["role"] == "user" else "Asistan"
                    )
                    first_line = msg["content"].split("\n")[0]
                    if len(first_line) > 200:
                        first_line = first_line[:200] + "…"
                    summary_lines.append(f"{role_label}: {first_line}")

                history_text = "\n".join(summary_lines)

                summary_input = build_styled_input(
                    f"Kullanıcı: {user_addressing}\n\n"
                    f"Geçmiş Sohbet:\n{history_text}\n\n"
                    f"Kullanıcı sorusu: {soru}\n\n"
                    f"Görev: Yukarıdaki geçmiş sohbeti kullanıcıya kısa, "
                    f"madde madde ve sıcak bir şekilde özetle. "
                    f"Hangi konular konuşuldu, hangi tur türleri önerildi "
                    f"ID değerleri ile beraber özetle. "
                    f"Nelerin üzerinde duruldu kısaca belirt. "
                    f"Gerekirse devam edebilecekleri konuları sor. "
                    f"Kullanıcıyı isimiyle hitap ederek yaz.",
                    ai_prompt_content,
                )

                response_result = await response_agent.run(summary_input, [])

                if isinstance(response_result, dict):
                    usage = response_result.get("usage")
                    if usage:
                        total_usage.add(usage)
                    response_content = response_result.get("text", "")
                else:
                    response_content = str(response_result)

                logger.info(
                    "📋 chat_summary üretildi (%d karakter)",
                    len(response_content),
                )

        # ─────────────────────────────────────────────────────────────
        # GENERAL CONVERSATION / OUT OF SCOPE
        # ─────────────────────────────────────────────────────────────
        elif any(
            label in labels
            for label in ["general_conversation", "out_of_scope"]
        ):
            response_result = await response_agent.run(
                response_input, chat_history
            )

            if isinstance(response_result, dict):
                usage = response_result.get("usage")
                if usage:
                    total_usage.add(usage)
                response_content = response_result.get("text", "")
            else:
                response_content = str(response_result)

        # ─────────────────────────────────────────────────────────────
        # FALLBACK
        # ─────────────────────────────────────────────────────────────
        else:
            if intent_result and isinstance(intent_result, dict):
                response_content = intent_result.get("text", "")
            else:
                response_content = (
                    f"{user_addressing}, sorunuzu anlayamadım. "
                    f"Lütfen biraz daha detay verir misiniz? 😊"
                )

        # =============================================================
        # 🔥 TOKEN HESAPLA & RESPONSE
        # =============================================================
        total_tokens = total_usage.total_tokens

        response_data = ResponseData(
            companyId=request.companyId,
            chatId=request.chatId,
            label=labels,
            content=response_content,
            model=model_llm,
            tokens_used=total_tokens,
        )

        return SocialResponse(success=True, data=response_data)

    except Exception as e:
        logger.exception(
            "Request işlenirken hata oluştu (chatId: %s)", request.chatId
        )
        return SocialResponse(success=False, errorMessage=str(e))
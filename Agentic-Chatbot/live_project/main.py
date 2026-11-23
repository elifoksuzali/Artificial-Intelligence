from response_structure import (
    ResponseData, 
    SocialRequest,
    SocialResponse,
    TourListItem,
    TourPayload,
    QuestionFieldResult,
)

# Ollama/vLLM agent'ları (free modda kullanılır)
from agentic_service import (
    main_router_agent as ollama_router,
    question_field_agent as ollama_field,
    tour_information_agent as ollama_tour_info,
    response_agent as ollama_response,
    summary_agent as ollama_summary,
    tour_id_resolver_agent as ollama_resolver,
    qdrant_response_agent as ollama_qdrant_response,
    reservation_info_agent as ollama_reservation,
)
from token_price import Usage, count_tokens
import json, logging, re, asyncio

from connection_qdrant import qdrant_host, qdrant_api_key

from setting import QDRANT_COLLECTION

from filter_search import (
    FILTER_SEARCH_FIELDS, 
    MULTIVECTOR_SEARCH_FIELDS,
    build_advanced_filter, 
    build_raw_filter, 
    resolve_payload_fields,
    classify_search_fields,
    build_kume_filter_dict,
    flatten_tour_payload,
    to_toon,
    extract_shown_tour_ids,
    find_original_tour_query,
)
from tool_calls import (
    call_mng_tour_program,
    calculate_rooms,
    tatil_ucreti_hesapla,
    EXTRA_ROOM_COST,
    _tf_to_label,
    _tf_to_value,
    _generate_voucher_data,
    send_voucher_email,
)
from multivector_search import (
    async_hybrid_search,
    async_filtered_hybrid_search,
    async_dense_search_with_filter,  # Basit: client.search(query_vector, query_filter)
    rerank_results
)
import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from connection_sql import KeyManager
key_manager = KeyManager()
api_list = key_manager.load_api_key()
model_llm = api_list.get('model_ollama', '')
model_gpt = api_list.get('model_gpt', 'gpt-4.1')

from constants import ErrorCode
from error_handler import APIError
MAX_CONCURRENT = 7
# Özetleme sonrası korunacak son mesaj sayısı
RECENT_MESSAGES_KEEP = 6

TOKEN_THRESHOLD = 5_000
# =========================================================
# region DYNAMIC AGENT SELECTION


# OpenAI agent'ları lazy import (sadece pro kullanıldığında yüklenir)
_openai_agents = None


def get_openai_agents() -> dict:
    """
    OpenAI agent'larını lazy load eder.
    İlk çağrıda import edilir, sonraki çağrılarda cache'den döner.
    """
    global _openai_agents
    if _openai_agents is None:
        try:
            from gpt_agents.agentic_service import (
                main_router_agent,
                question_field_agent,
                tour_information_agent,
                response_agent,
                summary_agent,
                tour_id_resolver_agent,
                qdrant_response_agent,
                reservation_info_agent,
            )
            _openai_agents = {
                "router": main_router_agent,
                "field": question_field_agent,
                "tour_info": tour_information_agent,
                "response": response_agent,
                "summary": summary_agent,
                "resolver": tour_id_resolver_agent,
                "qdrant_response": qdrant_response_agent,
                "reservation": reservation_info_agent,
            }
            logger.info("✅ OpenAI agent'ları yüklendi")
        except ImportError as e:
            logger.error(f"❌ OpenAI agent'ları yüklenemedi: {e}")
            raise
    return _openai_agents


def get_ollama_agents() -> dict:
    """Ollama/vLLM agent'larını döner."""
    return {
        "router": ollama_router,
        "field": ollama_field,
        "tour_info": ollama_tour_info,
        "response": ollama_response,
        "summary": ollama_summary,
        "resolver": ollama_resolver,
        "qdrant_response": ollama_qdrant_response,
        "reservation": ollama_reservation,
    }


def get_agents(contract_type: str) -> tuple[dict, str]:
    """
    Contract type'a göre doğru agent setini ve model adını döner.
    
    Args:
        contract_type: "free" (Ollama) veya "pro" (OpenAI)
    
    Returns:
        tuple: (agent_dict, model_name)
    """
    if contract_type == "pro":
        return get_openai_agents(), model_gpt
    else:  # free veya diğer
        return get_ollama_agents(), model_llm

# endregion
# =========================================================

# =========================================================
# region SMART MEMORY MANAGEMENT
# =========================================================

async def smart_memory_management(
    chat_history: list[dict],
    total_usage: Usage,
    summary_agent,  # Dinamik agent (Ollama veya OpenAI)
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
# endregion
# =========================================================

# =========================================================
# region STYLE APPLICATOR (aiSettings)
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
    tour_id_resolver_agent,  # Dinamik agent (Ollama veya OpenAI)
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

    # ── Sıra numarası kontrolü: "4. tur" → position 4 ──
    # (Not: "bu tur", "o tur" gibi referanslar artık LLM tarafından çözülüyor)
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

# endregion
# =========================================================

# =========================================================
# region ANA ROUTER FUNCTION
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

        # =============================================================
        # 📌 DİNAMİK AGENT SEÇİMİ
        # free → Ollama/vLLM | pro → OpenAI
        # =============================================================
        if contract_type not in ("free", "pro"):
            return SocialResponse(
                success=False,
                errorMessage=f"Bilinmeyen Sözleşme Türü!!! Paket: {contract_type}. "
                            f"Geçerli değerler: 'free', 'pro'",
            )

        try:
            agents, current_model = get_agents(contract_type)
            logger.info(f"🤖 Seçilen model: {current_model} ({contract_type})")
        except Exception as e:
            logger.error(f"Agent yükleme hatası: {e}")
            return SocialResponse(
                success=False,
                errorMessage=f"Agent yüklenemedi: {str(e)}",
            )

        # Agent'ları local değişkenlere ata
        main_router_agent = agents["router"]
        question_field_agent = agents["field"]
        tour_information_agent = agents["tour_info"]
        response_agent = agents["response"]
        summary_agent = agents["summary"]
        tour_id_resolver_agent = agents["resolver"]
        qdrant_response_agent = agents["qdrant_response"]
        reservation_info_agent = agents["reservation"]

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
            chat_history, total_usage, summary_agent
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

        raw_label = intent_result.get("types", ["unknown"]) if isinstance(intent_result, dict) else ["unknown"]
        VALID_LABELS = [
            "general_conversation",
            "chat_summary",
            "out_of_scope",
            "tour_list",
            "more_tours",
            "tour_info",
            "tour_program_detail",
            "price_calculate",
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
            # 🔹 ALAN SINIFLANDIRMASI: Filter vs Multivector
            # ---------------------------------------------------------
            filter_fields, multivector_fields, search_strategy = classify_search_fields(
                label_field, kume_list
            )
            
            logger.info(f"🔍 Filter alanları        : {filter_fields}")
            logger.info(f"🔍 Multivector alanları   : {multivector_fields}")
            logger.info(f"🔍 Arama stratejisi       : {search_strategy}")

            # ---------------------------------------------------------
            # 🔹 filter_dict oluştur (filter_fields'tan)
            # ---------------------------------------------------------
            filter_dict = build_kume_filter_dict(filter_fields, kume_list)

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
            # 🔹 with_payload hesaplama (tüm stratejiler için ortak)
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # 🔹 SEARCH STRATEJİSİ UYGULA
            # ---------------------------------------------------------
            results_flat: list[dict] = []
            
            # ═══════════════════════════════════════════════════════════
            # STRATEJİ 1: FILTER_ONLY - Sadece filtreleme (scroll)
            # ═══════════════════════════════════════════════════════════
            if search_strategy == "filter_only":
                logger.info("🔹 FILTER_ONLY: Scroll ile filtreli arama yapılacak")

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

                # ── Pydantic → Flatten ──
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

            # ═══════════════════════════════════════════════════════════
            # STRATEJİ 2: MULTIVECTOR_ONLY - Sadece hybrid search + rerank
            # Soru direkt embedding'e çevrilir, Qdrant'ta aranır
            # ═══════════════════════════════════════════════════════════
            elif search_strategy == "multivector_only":
                logger.info("🔹 MULTIVECTOR_ONLY: Hybrid search + rerank yapılacak")
                
                # Soru direkt embedding'e çevrilir
                search_query = field_query
                
                logger.info(f"🔍 Hybrid search sorgusu: {search_query}")
                
                hybrid_results = await async_hybrid_search(
                    collection_name=QDRANT_COLLECTION,
                    query=search_query,
                    limit=30,
                    top_k=10
                )
                
                logger.info(f"✅ Hybrid search sonuç sayısı: {len(hybrid_results)}")
                
                # Sonuçları results_flat formatına dönüştür
                for r in hybrid_results:
                    results_flat.append({
                        "id": r.get("id"),
                        "isim": r.get("isim"),
                        "gecesayisi": r.get("gecesayisi"),
                        "turtipi": r.get("turtipi"),
                        "konaklama": r.get("konaklama"),
                    })

            # ═══════════════════════════════════════════════════════════
            # STRATEJİ 3: HYBRID_COMBINED - Hem vektör hem filtre
            # Basit yapı: client.search(query_vector, query_filter)
            # ═══════════════════════════════════════════════════════════
            elif search_strategy == "hybrid_combined":
                logger.info("🔹 HYBRID_COMBINED: Vektör + filtre birleşik arama yapılacak")
                
                # Qdrant Filter objesi oluştur
                qdrant_filter = build_advanced_filter(filter_dict)
                
                if qdrant_filter:
                    logger.info(
                        "📌 Qdrant Filter: %s",
                        json.dumps(filter_dict, indent=2, ensure_ascii=False, default=str)
                    )
                    
                    # Soru direkt embedding'e çevrilir
                    search_query = field_query
                    
                    logger.info(f"🔍 Combined search sorgusu: {search_query}")
                    
                    # ════════════════════════════════════════════════════
                    # 🔥 BASİT YAPI: client.search(query_vector, query_filter)
                    # Hem vektör benzerliği hem de exact match filtresi
                    # ════════════════════════════════════════════════════
                    raw_results = await async_dense_search_with_filter(
                        collection_name=QDRANT_COLLECTION,
                        query=search_query,
                        query_filter=qdrant_filter,
                        limit=20  # Rerank öncesi daha fazla al
                    )
                    
                    logger.info(f"✅ Dense + Filter search sonuç sayısı: {len(raw_results)}")
                    
                    # Rerank uygula (sonuçları skorla ve sırala)
                    if raw_results:
                        reranked = rerank_results(
                            query=search_query,
                            results=raw_results,
                            top_k=10
                        )
                        
                        # Sonuçları results_flat formatına dönüştür
                        for r in reranked:
                            results_flat.append({
                                "id": r.get("id"),
                                "isim": r.get("isim"),
                                "gecesayisi": r.get("gecesayisi"),
                                "turtipi": r.get("turtipi"),
                                "konaklama": r.get("konaklama"),
                            })
                    
                    logger.info(f"🏆 Rerank sonrası sonuç sayısı: {len(results_flat)}")
                    
                else:
                    # Filter oluşturulamadıysa sadece multivector yap
                    logger.warning("⚠️ Filter oluşturulamadı, multivector_only'ye düşülüyor")
                    
                    search_query = field_query
                    
                    hybrid_results = await async_hybrid_search(
                        collection_name=QDRANT_COLLECTION,
                        query=search_query,
                        limit=30,
                        top_k=10
                    )
                    
                    for r in hybrid_results:
                        results_flat.append({
                            "id": r.get("id"),
                            "isim": r.get("isim"),
                            "gecesayisi": r.get("gecesayisi"),
                            "turtipi": r.get("turtipi"),
                            "konaklama": r.get("konaklama"),
                        })

            # ---------------------------------------------------------
            # 🔹 SONUÇLARI İŞLE VE YANIT OLUŞTUR
            # ---------------------------------------------------------
            toon_output = to_toon(results_flat)
            logger.info(
                "📋 TOON çıktı (%d tur):\n%s",
                len(results_flat),
                toon_output,
            )

            # ── TOON verilerini doğal dile çevir (qdrant_response_agent) ──
            if results_flat:
                tur_sayisi = len(results_flat)
                qdrant_resp_input = build_styled_input(
                    f"Kullanıcı: {user_addressing}\n"
                    f"Qdrant Sonuçları (TOON format):\n{toon_output}\n\n"
                    f"⚠️ ZORUNLU: Yukarıda TOPLAM {tur_sayisi} tur var. "
                    f"Yanıtında {tur_sayisi} turun HEPSİNİ listele, hiçbirini atlama!\n\n"
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
                    soru, chat_history, total_usage, tour_id_resolver_agent
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
                    soru, chat_history, total_usage, tour_id_resolver_agent
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
                    except Exception as e:
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
        # PRICE CALCULATE — Konuşma tabanlı adım adım fiyat hesaplama
        # Agent soru-cevap ile tarih, kişi, yaş bilgisi toplar.
        # Tüm bilgiler tamamlandığında oda hesabı + fiyat üretir.
        # Oda kuralı: 1 oda = 2Y+1Ç veya 2Y+1Ç+1Bebek(≤3) veya 3Y
        # İlave oda ücreti: EXTRA_ROOM_COST (2000)
        # ─────────────────────────────────────────────────────────────
        elif "price_calculate" in labels:

            logger.info("💰 price_calculate handler başlatıldı")

            # ── 1. Tur ID belirle ──
            tour_id = None
            tour_name = ""

            # label_field'den direkt ID
            field_result = await question_field_agent.run(soru, chat_history)
            if isinstance(field_result, dict):
                usage = field_result.get("usage")
                if usage:
                    total_usage.add(usage)
                lf = field_result.get("label_field", {})
                if lf.get("id"):
                    tour_id = int(lf["id"])

            # Fuzzy resolution
            if tour_id is None:
                tour_resolved = await resolve_tour_id(
                    soru, chat_history, total_usage, tour_id_resolver_agent
                )
                if tour_resolved:
                    tour_id = tour_resolved["tour_id"]
                    tour_name = tour_resolved.get("tour_name", "")
                    logger.info(
                        "🎯 price_calculate: Fuzzy çözüm → ID=%d (%s)",
                        tour_id, tour_name,
                    )

            # ── Fallback: Devam eden rezervasyon akışından tur ID'si ──
            # "5 ve 9 yaşında" gibi devam mesajlarında tur referansı
            # olmayacağından geçmişteki en son konuşulan turu kullan
            if tour_id is None:
                last_tour = _find_last_mentioned_tour(chat_history)
                if last_tour:
                    tour_id = last_tour["tour_id"]
                    tour_name = last_tour.get("tour_name", "")
                    logger.info(
                        "🎯 price_calculate: Devam eden akıştan "
                        "ID=%d (%s)",
                        tour_id, tour_name,
                    )

            if tour_id is None:
                response_content = (
                    f"{user_addressing}, fiyat hesaplamak için hangi turu "
                    f"seçmek istediğinizi belirtir misiniz? "
                    f"Tur numarası veya adıyla söyleyebilirsiniz. 😊"
                )
            else:
                # ── 2. Qdrant'tan tarihFiyat çek ──
                id_filter = {
                    "must": [{"key": "id", "match": {"value": tour_id}}]
                }
                scroll_body = {
                    "filter": id_filter,
                    "limit": 1,
                    "with_payload": ["id", "isim", "tarihFiyat"],
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
                    tour_name = payload.get("isim", tour_name)
                    tarih_fiyat_list = payload.get("tarihFiyat", []) or []

                    if tarih_fiyat_list:
                        # ── 3. Konuşmadan rezervasyon bilgisi çıkar ──
                        available_dates_text = "\n".join([
                            f"- {_tf_to_label(tf)} ({tf.get('kisibasi', 0)} "
                            f"{tf.get('kur', 'TL')}, Kalan: {tf.get('kalan', 0)})"
                            for tf in tarih_fiyat_list
                        ])

                        recent_chat = chat_history[-8:] if chat_history else []
                        chat_context = "\n".join([
                            f"{m['role']}: {m['content'][:300]}"
                            for m in recent_chat
                        ])

                        extract_input = (
                            f"Müsait tarihler:\n{available_dates_text}\n\n"
                            f"Konuşma bağlamı:\n{chat_context}\n\n"
                            f"Son mesaj: {soru}"
                        )

                        extract_result = await reservation_info_agent.run(
                            extract_input, []
                        )

                        # Parse extraction result
                        reservation_info = {}
                        if isinstance(extract_result, dict):
                            usage = extract_result.get("usage")
                            if usage:
                                total_usage.add(usage)
                            # JSON olarak dönmüş olabilir
                            if "tarih" in extract_result:
                                reservation_info = extract_result
                            else:
                                raw_text = extract_result.get("text", "")
                                try:
                                    reservation_info = json.loads(raw_text)
                                except (json.JSONDecodeError, ValueError):
                                    reservation_info = {}

                        logger.info(
                            "📋 Rezervasyon bilgileri: %s", reservation_info
                        )

                        selected_date_str = reservation_info.get("tarih")
                        yetiskin = reservation_info.get("yetiskin")
                        cocuk = reservation_info.get("cocuk", 0) or 0
                        cocuk_yaslari = (
                            reservation_info.get("cocuk_yaslari", []) or []
                        )
                        # LLM tabanlı onay ve ödeme tespiti
                        onay = reservation_info.get("onay", False)
                        odeme_istegi = reservation_info.get("odeme_istegi", False)

                        # ────────────────────────────────────────────
                        # ÖDEME LİNKİ — Kullanıcı ödeme linki istiyorsa
                        # ────────────────────────────────────────────
                        if odeme_istegi:
                            logger.info(
                                "💳 Ödeme linki talebi tespit edildi (LLM)"
                            )
                            payment_link = (
                                "https://payment.example.com/"
                                "?i=OTEvQ2srbnZWazdwYmphOG1tQ3BOdz09"
                            )
                            response_content = (
                                f"{user_addressing}, işte ödeme linkiniz! 💳\n\n"
                                f"🔗 **Güvenli Ödeme Sayfası:**\n"
                                f"{payment_link}\n\n"
                                f"Bu link üzerinden kredi kartı ile "
                                f"güvenli bir şekilde ödemenizi "
                                f"tamamlayabilirsiniz. 🔒\n\n"
                                f"Ödemeniz onaylandığında size "
                                f"e-posta ile bilgilendirme yapılacaktır. "
                                f"📧✨\n\n"
                                f"İyi tatiller dileriz! 🌴🎉"
                            )

                        # ────────────────────────────────────────────
                        # ONAY TESPİTİ — Kullanıcı fiyatı onayladıysa
                        # voucher oluştur
                        # ────────────────────────────────────────────
                        elif onay and selected_date_str and yetiskin:
                            logger.info(
                                "✅ Kullanıcı onayı tespit edildi (LLM)"
                            )
                            # Tarihi tarihFiyat'tan eşleştir
                            selected_tf = None
                            ds_lower = selected_date_str.lower()
                            for tf in tarih_fiyat_list:
                                label = _tf_to_label(tf)
                                if (
                                    ds_lower in label.lower()
                                    or label.lower() in ds_lower
                                ):
                                    selected_tf = tf
                                    break

                            if selected_tf:
                                kisibasi = selected_tf.get("kisibasi", 0)
                                kur = selected_tf.get("kur", "TL")
                                tarih_label = _tf_to_label(selected_tf)

                                # Oda ve fiyat hesapla
                                oda_sayisi, ekstra_oda, _ = calculate_rooms(
                                    yetiskin, cocuk_yaslari
                                )
                                oda_basi, cocuk_detay = tatil_ucreti_hesapla(
                                    yetiskin, kisibasi, cocuk_yaslari
                                )
                                ekstra_oda_ucreti = ekstra_oda * EXTRA_ROOM_COST
                                genel_toplam = oda_basi + ekstra_oda_ucreti

                                # Müşteri bilgilerini al
                                customer_dict = {
                                    "firstName": request.customer.firstName,
                                    "lastName": request.customer.lastName,
                                    "phone": request.customer.phone,
                                    "email": request.customer.email,
                                    "gender": request.customer.gender,
                                }

                                # Voucher verisi oluştur
                                voucher_data = _generate_voucher_data(
                                    company_id=request.companyId,
                                    customer=customer_dict,
                                    tour_id=tour_id,
                                    tour_name=tour_name,
                                    tarih=tarih_label,
                                    oda_sayisi=oda_sayisi,
                                    yetiskin=yetiskin,
                                    cocuk=cocuk,
                                    cocuk_yaslari=cocuk_yaslari,
                                    kisibasi=kisibasi,
                                    genel_toplam=genel_toplam,
                                    kur=kur,
                                )

                                # E-posta gönder
                                customer_email = voucher_data['musteri']['email']
                                send_voucher_email(customer_email, voucher_data)

                                # Sade yanıt (e-posta gönderildi bilgisi)
                                response_content = (
                                    f"{user_addressing}, rezervasyonunuz "
                                    f"başarıyla oluşturuldu! 🎉🎊\n\n"
                                    f"📋 **Rezervasyon No:** "
                                    f"{voucher_data['rezervasyon_no']}\n"
                                    f"✉️ **E-posta:** {customer_email}\n\n"
                                    f"✅ Voucher bilgileriniz gönderilmiştir!\n\n"
                                    f"Mailinizi kontrol edin, eğer doğru ise "
                                    f"💳 Ödeme işleminizi tamamlamak için "
                                    f"size bir ödeme linki gönderebilirim 😊"
                                )

                                logger.info(
                                    "✅ Rezervasyon oluşturuldu: %s | E-posta: %s",
                                    voucher_data["rezervasyon_no"],
                                    customer_email,
                                )

                        # ────────────────────────────────────────────
                        # ADIM 1: Tarih ve kişi sayısı henüz yok → Sor
                        # ────────────────────────────────────────────
                        elif not selected_date_str or yetiskin is None:
                            dates_display = "\n".join([
                                f"  📅 {_tf_to_label(tf)} — "
                                f"{tf.get('kisibasi', 0):,.0f} "
                                f"{tf.get('kur', 'TL')}"
                                for tf in tarih_fiyat_list
                            ])
                            response_content = (
                                f"{user_addressing}, [#{tour_id}] "
                                f"{tour_name} turu için "
                                f"rezervasyon sürecini başlatalım! 😊\n\n"
                                f"📅 Müsait tarihlerimiz:\n"
                                f"{dates_display}\n\n"
                                f"Hangi tarihte katılmak istiyorsunuz "
                                f"ve kaç kişisiniz? "
                                f"(yetişkin ve varsa çocuk sayısını "
                                f"belirtiniz) 🎒"
                            )

                        # ────────────────────────────────────────────
                        # ADIM 2: Çocuk var ama yaşları eksik → Sor
                        # ────────────────────────────────────────────
                        elif cocuk > 0 and len(cocuk_yaslari) < cocuk:
                            response_content = (
                                f"{user_addressing}, {selected_date_str} "
                                f"tarihinde {yetiskin} yetişkin + "
                                f"{cocuk} çocuk olarak not ettim. ✍️\n\n"
                                f"Çocuklarınızın yaşlarını öğrenebilir "
                                f"miyim? (Yaş aralığına göre "
                                f"fiyatlandırma değişmektedir)\n\n"
                                f"💡 Çocuk ücret kuralları:\n"
                                f"  • 0-6 yaş: Ücretsiz\n"
                                f"  • 7-12 yaş: %50 indirimli\n"
                                f"  • 13 yaş ve üzeri: Yetişkin ücreti 😊"
                            )

                        # ────────────────────────────────────────────
                        # ADIM 3: Tüm bilgiler tamam → Hesapla
                        # ────────────────────────────────────────────
                        else:
                            # Tarihi tarihFiyat listesinden eşleştir
                            selected_tf = None
                            if selected_date_str:
                                ds_lower = selected_date_str.lower()
                                for tf in tarih_fiyat_list:
                                    label = _tf_to_label(tf)
                                    val = _tf_to_value(tf)
                                    if (
                                        ds_lower in label.lower()
                                        or label.lower() in ds_lower
                                        or val == selected_date_str
                                    ):
                                        selected_tf = tf
                                        break

                                # Fuzzy: kelime bazlı eşleşme
                                if selected_tf is None:
                                    words = [
                                        w for w in ds_lower.split()
                                        if len(w) > 2
                                    ]
                                    for tf in tarih_fiyat_list:
                                        lbl = _tf_to_label(tf).lower()
                                        if any(w in lbl for w in words):
                                            selected_tf = tf
                                            break

                            if selected_tf is None:
                                dates_disp = "\n".join([
                                    f"  📅 {_tf_to_label(tf)} — "
                                    f"{tf.get('kisibasi', 0):,.0f} "
                                    f"{tf.get('kur', 'TL')}"
                                    for tf in tarih_fiyat_list
                                ])
                                response_content = (
                                    f"{user_addressing}, seçtiğiniz tarih "
                                    f"({selected_date_str}) uygun değil. "
                                    f"😕\n\nMüsait tarihlerimiz:\n"
                                    f"{dates_disp}\n\n"
                                    f"Lütfen yukarıdaki tarihlerden "
                                    f"birini seçiniz."
                                )
                            else:
                                kisibasi = selected_tf.get("kisibasi", 0)
                                kur = selected_tf.get("kur", "TL")
                                kalan = selected_tf.get("kalan", 0)
                                tarih_label = _tf_to_label(selected_tf)

                                # ── Oda hesapla ──
                                oda_sayisi, ekstra_oda, oda_aciklama = (
                                    calculate_rooms(yetiskin, cocuk_yaslari)
                                )

                                # ── Kontenjan kontrolü ──
                                toplam_kisi = yetiskin + cocuk
                                if kalan > 0 and toplam_kisi > kalan:
                                    alts = []
                                    for tf in tarih_fiyat_list:
                                        tf_k = tf.get("kalan", 0)
                                        if (
                                            tf_k >= toplam_kisi
                                            and _tf_to_value(tf)
                                            != _tf_to_value(selected_tf)
                                        ):
                                            alts.append(
                                                f"  📅 {_tf_to_label(tf)} "
                                                f"(Kalan: {tf_k} kişi, "
                                                f"{tf.get('kisibasi', 0):,.0f}"
                                                f" {tf.get('kur', 'TL')})"
                                            )

                                    alt_text = ""
                                    if alts:
                                        alt_text = (
                                            f"\n\n📌 Yeterli kontenjanlı "
                                            f"alternatif tarihler:\n"
                                            + "\n".join(alts)
                                        )
                                    else:
                                        alt_text = (
                                            "\n\n😔 Maalesef yeterli "
                                            "kontenjanlı başka tarih "
                                            "bulunmamaktadır."
                                        )

                                    response_content = (
                                        f"{user_addressing}, {tarih_label} "
                                        f"tarihinde yeterli kontenjan "
                                        f"bulunmamaktadır. "
                                        f"(Kalan: {kalan} kişi, talep: "
                                        f"{toplam_kisi} kişi)"
                                        f"{alt_text}\n\n"
                                        f"Başka bir tarih seçmek ister "
                                        f"misiniz? 😊"
                                    )
                                else:
                                    # ── Fiyat hesapla ──
                                    oda_basi, cocuk_detay = (
                                        tatil_ucreti_hesapla(
                                            yetiskin, kisibasi,
                                            cocuk_yaslari,
                                        )
                                    )
                                    yetiskin_toplam = yetiskin * kisibasi
                                    cocuk_toplam = sum(
                                        c["ucret"] for c in cocuk_detay
                                    )
                                    ekstra_oda_ucreti = (
                                        ekstra_oda * EXTRA_ROOM_COST
                                    )
                                    genel_toplam = (
                                        oda_basi + ekstra_oda_ucreti
                                    )

                                    # Sade fiyat mesajı
                                    response_content = (
                                        f"{user_addressing}, [#{tour_id}] "
                                        f"{tour_name} turu için "
                                        f"**{tarih_label}** tarihinde "
                                        f"**{yetiskin} yetişkin"
                                    )
                                    if cocuk:
                                        response_content += (
                                            f" + {cocuk} çocuk"
                                        )
                                    response_content += (
                                        f"** için toplam fiyat:\n\n"
                                        f"💰 **{genel_toplam:,.0f} {kur}**\n\n"
                                    )
                                    if ekstra_oda > 0:
                                        response_content += (
                                            f"_(İlave oda ücreti dahil: "
                                            f"{ekstra_oda_ucreti:,.0f} {kur})_\n\n"
                                        )
                                    response_content += (
                                        f"Bu fiyat sizin için uygunsa "
                                        f"ön rezervasyonunuzu oluşturalım, "
                                        f"ardından voucher bilgilerinizi "
                                        f"paylaşacağım. ✨\n\n"
                                        f"Onaylıyor musunuz? 😊"
                                    )

                                    logger.info(
                                        "✅ Fiyat hesaplandı: "
                                        "tour=%d, tarih=%s, "
                                        "toplam=%.0f %s, "
                                        "oda=%d (ekstra=%d)",
                                        tour_id, tarih_label,
                                        genel_toplam, kur,
                                        oda_sayisi, ekstra_oda,
                                    )
                    else:
                        response_content = (
                            f"{user_addressing}, [#{tour_id}] {tour_name} "
                            f"için şu an fiyat bilgisi bulunmamaktadır. 😔"
                        )
                else:
                    response_content = (
                        f"{user_addressing}, {tour_id} numaralı tur "
                        f"bulunamadı. Farklı bir tur belirtir misiniz? 😊"
                    )

            logger.info("✅ price_calculate tamamlandı")

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
            model=current_model,
            tokens_used=total_tokens,
        )

        return SocialResponse(success=True, data=response_data)

    except APIError as e:
        # Özel API hataları (error code ile)
        logger.error(
            f"[{e.code}] API Hatası (chatId: {request.chatId}): "
            f"{e.message} - {e.detail}"
        )
        return SocialResponse(
            success=False,
            errorMessage=f"[{e.code}] {e.message}: {e.detail}"
        )
        
    except httpx.TimeoutException as e:
        error_code = ErrorCode.LLM_TIMEOUT
        logger.error(
            f"[{error_code.code}] Timeout hatası (chatId: {request.chatId}): {e}"
        )
        return SocialResponse(
            success=False,
            errorMessage=f"[{error_code.code}] {error_code.message}"
        )
        
    except httpx.ConnectError as e:
        error_code = ErrorCode.LLM_CONNECTION_ERROR
        logger.error(
            f"[{error_code.code}] Bağlantı hatası (chatId: {request.chatId}): {e}"
        )
        return SocialResponse(
            success=False,
            errorMessage=f"[{error_code.code}] {error_code.message}"
        )
        
    except json.JSONDecodeError as e:
        error_code = ErrorCode.LLM_RESPONSE_PARSE_ERROR
        logger.error(
            f"[{error_code.code}] JSON parse hatası (chatId: {request.chatId}): {e}"
        )
        return SocialResponse(
            success=False,
            errorMessage=f"[{error_code.code}] {error_code.message}: {str(e)[:100]}"
        )
        
    except Exception as e:
        error_code = ErrorCode.UNKNOWN_ERROR
        logger.exception(
            f"[{error_code.code}] Beklenmeyen hata (chatId: {request.chatId}): "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        return SocialResponse(
            success=False,
            errorMessage=f"[{error_code.code}] {error_code.message}: {type(e).__name__}"
        )
        
# endregion
# =========================================================

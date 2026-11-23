import os

from response_structure import (
    PriceCalculateRequest,
    PriceCalculateResponse,
    AlternatifTarih
)
import httpx,asyncio
from connection_qdrant import QdrantDatabase, qdrant_host, qdrant_api_key
from setting import QDRANT_COLLECTION
import re
from datetime import datetime
import uuid
import math
import logging

logger = logging.getLogger(__name__)
from constants import ErrorCode
import traceback
import smtplib
from email.mime.text import MIMEText

async def calculate_price(
    calc_request: PriceCalculateRequest,
) -> PriceCalculateResponse:
    """
    STATE-MACHINE tabanlı fiyat hesaplama.

    Adımlar (step):
        tour_id  → Qdrant'tan tur bilgisini çek
        date     → Seçilen tarihi doğrula, yoksa yakın tarihleri öner
        rooms    → Oda kapasitesini kontrol et (max 3 kişi/oda), gerekirse güncelle
        quota    → Kontenjan kontrolü (kalan kişi ≥ toplam kişi)
        pricing  → Fiyatı hesapla
        completed → Sonuç
    """

    # =========================================================
    # STEP 1 — TOUR_ID : Qdrant'tan tur bilgisi çek
    # =========================================================
    id_filter = {"must": [{"key": "id", "match": {"value": calc_request.tour_id}}]}
    scroll_body = {
        "filter": id_filter,
        "limit": 1,
        "with_payload": ["id", "isim", "tarihFiyat"],
        "with_vector": False,
    }

    async with httpx.AsyncClient(verify=False) as http_qdrant:
        qdrant_resp = await http_qdrant.post(
            f"{qdrant_host}/collections/{QDRANT_COLLECTION}/points/scroll",
            json=scroll_body,
            headers={"api-key": qdrant_api_key, "Content-Type": "application/json"},
            timeout=60,
        )
        qdrant_data = qdrant_resp.json()

    raw_points = qdrant_data.get("result", {}).get("points", [])
    if not raw_points:
        return _make_error_response(calc_request, "tour_id", "Tur bulunamadı.")

    payload = raw_points[0].get("payload", {})
    tour_name = payload.get("isim", "")
    tarih_fiyat_list: list[dict] = payload.get("tarihFiyat", []) or []

    if not tarih_fiyat_list:
        return _make_error_response(
            calc_request, "tour_id",
            f"[#{calc_request.tour_id}] {tour_name} için fiyat bilgisi bulunamadı.",
            tour_name=tour_name,
        )

    # Tüm müsait tarihleri indexle
    available_dates: dict[str, dict] = {}
    for tf in tarih_fiyat_list:
        available_dates[_tf_to_value(tf)] = tf

    logger.info(
        "📅 Müsait tarihler (%d adet): %s",
        len(available_dates), list(available_dates.keys()),
    )

    # =========================================================
    # STEP 2 — DATE : Seçilen tarihi doğrula
    # =========================================================
    selected_tf = available_dates.get(calc_request.tur_tarihi)

    if selected_tf is None:
        # En yakın tarihleri öner
        all_alts = [_tf_to_alternatif(tf) for tf in tarih_fiyat_list]
        return _make_error_response(
            calc_request, "date",
            f"Seçilen tarih ({calc_request.tur_tarihi}) uygun değil. "
            f"Lütfen aşağıdaki {len(all_alts)} müsait tarihten birini seçiniz.",
            tour_name=tour_name,
            alternatif_tarihler=all_alts,
        )

    kisibasi: float = selected_tf.get("kisibasi", 0)
    kur: str = selected_tf.get("kur", "TL")
    kalan: int = selected_tf.get("kalan", 0)
    tarih_label = _tf_to_label(selected_tf)

    # =========================================================
    # STEP 3 — ROOMS : Oda kapasitesi kontrolü
    # Kural: 2Y+1Ç veya 2Y+1Ç+1Bebek(≤3) veya 3Y = 1 oda
    # İlave oda ücreti: EXTRA_ROOM_COST
    # =========================================================
    toplam_kisi = calc_request.yetiskin_sayisi + calc_request.cocuk_sayisi
    gereken_oda, ekstra_oda, oda_aciklama = calculate_rooms(
        calc_request.yetiskin_sayisi, calc_request.cocuk_yaslari
    )
    oda_sayisi = max(calc_request.oda_sayisi, gereken_oda)
    oda_guncellendi = oda_sayisi > calc_request.oda_sayisi
    oda_uyari = oda_aciklama if oda_guncellendi else ""
    ekstra_oda = max(0, oda_sayisi - 1)

    if oda_guncellendi:
        logger.info(
            "🏨 Oda güncellendi: %d → %d (%s)",
            calc_request.oda_sayisi, oda_sayisi, oda_uyari,
        )

    # =========================================================
    # STEP 4 — QUOTA : Kontenjan kontrolü
    # =========================================================
    kontenjan_pirlekli = False
    alternatif_tarihler: list[AlternatifTarih] = []

    if kalan > 0 and toplam_kisi > kalan:
        kontenjan_pirlekli = True
        # Yeterli kontenjanlı alternatif tarihleri bul
        for tf in tarih_fiyat_list:
            tf_kalan = tf.get("kalan", 0)
            if tf_kalan >= toplam_kisi and _tf_to_value(tf) != calc_request.tur_tarihi:
                alternatif_tarihler.append(_tf_to_alternatif(tf))

        alt_msg = ""
        if alternatif_tarihler:
            alt_msg = (
                f" Yeterli kontenjanlı {len(alternatif_tarihler)} alternatif tarih öneriyoruz."
            )
        else:
            alt_msg = " Maalesef yeterli kontenjanlı başka tarih bulunmamaktadır."

        return _make_error_response(
            calc_request, "quota",
            f"Bu tarihte ({tarih_label}) yeterli kontenjan yok. "
            f"Kalan: {kalan} kişi, talep: {toplam_kisi} kişi.{alt_msg}",
            tour_name=tour_name,
            tarih=tarih_label,
            kur=kur,
            kisibasi_fiyat=kisibasi,
            oda_sayisi=oda_sayisi,
            alternatif_tarihler=alternatif_tarihler,
        )

    # =========================================================
    # STEP 5 — PRICING : Fiyatı hesapla (ilave oda ücreti dahil)
    # =========================================================
    oda_basi_toplam, cocuk_detay = tatil_ucreti_hesapla(
        yetiskin_sayisi=calc_request.yetiskin_sayisi,
        yetiskin_fiyat=kisibasi,
        cocuk_yaslari=calc_request.cocuk_yaslari,
    )

    yetiskin_toplam = calc_request.yetiskin_sayisi * kisibasi
    cocuk_toplam = sum(c["ucret"] for c in cocuk_detay)
    ekstra_oda_ucreti = ekstra_oda * EXTRA_ROOM_COST
    genel_toplam = oda_basi_toplam + ekstra_oda_ucreti

    # =========================================================
    # STEP 6 — COMPLETED : Özet mesaj
    # =========================================================
    mesaj_parts = [
        f"=== REZERVASYON BİLGİLERİ ===",
        f"Tur ID: {calc_request.tour_id}",
        f"Tur: {tour_name}",
        f"Tarih: {tarih_label}",
        f"Oda Sayısı: {oda_sayisi}",
        f"Yetişkin: {calc_request.yetiskin_sayisi} kişi × {kisibasi:,.2f} = {yetiskin_toplam:,.2f} {kur}",
    ]

    if oda_guncellendi:
        mesaj_parts.append(f"⚠️ {oda_uyari}")

    if ekstra_oda > 0:
        mesaj_parts.append(
            f"💰 İlave oda ücreti: {ekstra_oda} × {EXTRA_ROOM_COST:,} = "
            f"{ekstra_oda_ucreti:,} {kur}"
        )

    if cocuk_detay:
        mesaj_parts.append(f"Çocuk ({len(cocuk_detay)} kişi):")
        for cd in cocuk_detay:
            mesaj_parts.append(
                f"  • {cd['yas']} yaş → {cd['indirim']} = {cd['ucret']:,.2f} {kur}"
            )
        yas_listesi = ", ".join(str(cd["yas"]) for cd in cocuk_detay)
        mesaj_parts.append(f"Çocuk Yaşları: [{yas_listesi}]")
        mesaj_parts.append(f"Çocuk Toplam: {cocuk_toplam:,.2f} {kur}")

    mesaj_parts.append(f"Kişi Bazlı Toplam: {oda_basi_toplam:,.2f} {kur}")

    if ekstra_oda > 0:
        mesaj_parts.append(
            f"Kişi Toplam + İlave Oda: {oda_basi_toplam:,.2f} + "
            f"{ekstra_oda_ucreti:,} = {genel_toplam:,.2f} {kur}"
        )

    mesaj_parts.append(f"TOPLAM FİYAT: {genel_toplam:,.2f} {kur}")

    if kalan > 0:
        mesaj_parts.append(f"Kalan Kontenjan: {kalan} kişi")

    logger.info(
        "✅ Fiyat hesaplandı: tour=%d, tarih=%s, toplam=%.2f %s, "
        "oda=%d (ekstra=%d)",
        calc_request.tour_id, tarih_label, genel_toplam, kur,
        oda_sayisi, ekstra_oda,
    )

    return PriceCalculateResponse(
        success=True,
        step="completed",
        tour_id=calc_request.tour_id,
        tour_name=tour_name,
        tarih=tarih_label,
        kur=kur,
        kisibasi_fiyat=kisibasi,
        yetiskin_sayisi=calc_request.yetiskin_sayisi,
        yetiskin_toplam=yetiskin_toplam,
        cocuk_detay=cocuk_detay,
        cocuk_toplam=cocuk_toplam,
        oda_sayisi=oda_sayisi,
        oda_guncellendi=oda_guncellendi,
        oda_uyari=oda_uyari,
        oda_basi_toplam=oda_basi_toplam,
        ekstra_oda_ucreti=ekstra_oda_ucreti,
        genel_toplam=genel_toplam,
        kalan_kontenjan=kalan,
        kontenjan_pirlekli=kontenjan_pirlekli,
        mesaj="\n".join(mesaj_parts),
    )


# =========================================================
# region FİYAT HESAPLAMA
# =========================================================

def tatil_ucreti_hesapla(yetiskin_sayisi: int, yetiskin_fiyat: float, cocuk_yaslari: list[int]) -> tuple[float, list[dict]]:
    """
    Tek oda için fiyat hesaplar.
    
    Kurallar:
        - 0-6 yaş: Ücretsiz
        - 7-12 yaş: %50 indirimli
        - 13+ yaş: Yetişkin ücreti
    
    Returns: (oda_toplam, cocuk_detay_listesi)
    """
    toplam_ucret = yetiskin_sayisi * yetiskin_fiyat
    cocuk_detay: list[dict] = []
    
    for yas in cocuk_yaslari:
        if yas <= 6:
            cocuk_ucreti = 0.0
            indirim = "Ücretsiz (0-6 yaş)"
        elif 7 <= yas <= 12:
            cocuk_ucreti = yetiskin_fiyat * 0.5
            indirim = "%50 indirimli (7-12 yaş)"
        else:
            cocuk_ucreti = yetiskin_fiyat
            indirim = "Tam ücret (13+ yaş)"
            
        toplam_ucret += cocuk_ucreti
        cocuk_detay.append({
            "yas": yas,
            "indirim": indirim,
            "ucret": cocuk_ucreti,
        })
        
    return toplam_ucret, cocuk_detay


_AY_MAP = {
    "Ocak": "01", "Şubat": "02", "Mart": "03", "Nisan": "04",
    "Mayıs": "05", "Haziran": "06", "Temmuz": "07", "Ağustos": "08",
    "Eylül": "09", "Ekim": "10", "Kasım": "11", "Aralık": "12",
}
MAX_PER_ROOM = 3       # Bir odada max sayılır kişi (yetişkin + 4+ yaş çocuk)
EXTRA_ROOM_COST = 2000 # İlave oda ücreti

# ── Rezervasyon onay tespiti (prompt tabanlı) ──
# Manuel keyword kontrolü KALDIRILDI — LLM prompt ile tespit edilir


def _extract_tour_duration(tour_name: str) -> str:
    """Tur adından süresini çıkar. Ör: '7 Gece' → '8 Gün / 7 Gece'"""
    match = re.search(r'(\d+)\s*[Gg]ece', tour_name)
    if match:
        gece = int(match.group(1))
        return f"{gece + 1} Gün / {gece} Gece"
    return "Belirtilmemiş"


def _generate_reservation_number() -> str:
    """Benzersiz rezervasyon numarası üret. Ör: MNG-2026-ABC123"""
    year = datetime.now().year
    unique_part = uuid.uuid4().hex[:6].upper()
    return f"MNG-{year}-{unique_part}"


def _generate_voucher_data(
    company_id: str,
    customer: dict,
    tour_id: int,
    tour_name: str,
    tarih: str,
    oda_sayisi: int,
    yetiskin: int,
    cocuk: int,
    cocuk_yaslari: list[int],
    kisibasi: float,
    genel_toplam: float,
    kur: str,
) -> dict:
    """Rezervasyon voucher verisi oluştur."""
    rezervasyon_no = _generate_reservation_number()
    tur_suresi = _extract_tour_duration(tour_name)
    
    return {
        "rezervasyon_no": rezervasyon_no,
        "olusturma_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "firma": {
            "ad": company_id,
        },
        "musteri": {
            "ad_soyad": f"{customer.get('firstName', '')} {customer.get('lastName', '')}".strip(),
            "telefon": customer.get("phone", ""),
            "email": customer.get("email", ""),
            "cinsiyet": "Kadın" if customer.get("gender", "").upper() == "F" else "Erkek",
        },
        "tur": {
            "kod": f"#{tour_id}",
            "ad": tour_name,
            "baslangic_tarihi": tarih,
            "sure": tur_suresi,
        },
        "konaklama": {
            "oda_sayisi": oda_sayisi,
            "yetiskin_sayisi": yetiskin,
            "cocuk_sayisi": cocuk,
            "cocuk_yaslari": cocuk_yaslari,
            "toplam_konuk": yetiskin + cocuk,
        },
        "fiyat": {
            "kisi_basi": kisibasi,
            "kur": kur,
            "toplam": genel_toplam,
        },
    }


def calculate_rooms(yetiskin: int, cocuk_yaslari: list[int]) -> tuple[int, int, str]:
    """
    Oda kapasitesi hesaplama — yeni kurallar:

    Geçerli tek-oda kombinasyonları:
        • 2 yetişkin + 1 çocuk (4+ yaş)
        • 2 yetişkin + 1 çocuk + 1 bebek (0-3 yaş) — bebek ek yatakta ücretsiz
        • 3 yetişkin
    Bunların dışında → ilave oda gerekir.

    Sayılır kişi = yetişkin + çocuk(4+ yaş)
    Her oda max 3 sayılır kişi.
    Her odaya ek olarak 1 bebek (0-3 yaş) ücretsiz kalabilir.

    Returns: (toplam_oda, ekstra_oda, aciklama)
    """
    bebekler = sum(1 for y in cocuk_yaslari if y <= 3)
    buyuk_cocuklar = sum(1 for y in cocuk_yaslari if y > 3)

    sayilir_kisi = yetiskin + buyuk_cocuklar

    # Her oda max 3 sayılır kişi
    oda = max(1, math.ceil(sayilir_kisi / MAX_PER_ROOM))

    # Bebek dağılımı: her odaya max 1 bebek
    if bebekler > oda:
        oda = bebekler  # En az bebek sayısı kadar oda lazım

    ekstra_oda = max(0, oda - 1)

    aciklama = ""
    if ekstra_oda > 0:
        aciklama = (
            f"1 odada max 2 yetişkin + 1 çocuk (veya 3 yetişkin) kalabilir. "
            f"Toplam {sayilir_kisi} sayılır kişi"
        )
        if bebekler:
            aciklama += f" + {bebekler} bebek (0-3 yaş)"
        aciklama += (
            f" için {oda} oda gereklidir. "
            f"İlave {ekstra_oda} oda ücreti ({ekstra_oda} × {EXTRA_ROOM_COST:,} = "
            f"{ekstra_oda * EXTRA_ROOM_COST:,}) uygulanmıştır."
        )

    return oda, ekstra_oda, aciklama


def _tf_to_value(tf: dict) -> str:
    """tarihFiyat dict → 'YYYY-MM-DD' string"""
    tarih = tf.get("tarih", {}) or {}
    gun = tarih.get("gun", 1)
    ay = tarih.get("ay", "")
    yil = tarih.get("yil", 2025)
    ay_num = _AY_MAP.get(ay, "01")
    return f"{yil}-{ay_num}-{str(gun).zfill(2)}"


def _tf_to_label(tf: dict) -> str:
    """tarihFiyat dict → '27 Eylül 2025' label"""
    tarih = tf.get("tarih", {}) or {}
    return f"{tarih.get('gun', '')} {tarih.get('ay', '')} {tarih.get('yil', '')}".strip()


def _tf_to_alternatif(tf: dict) -> AlternatifTarih:
    return AlternatifTarih(
        value=_tf_to_value(tf),
        label=_tf_to_label(tf),
        kisibasi=tf.get("kisibasi", 0.0),
        kur=tf.get("kur", "TL"),
        kalan=tf.get("kalan", 0),
    )


def _make_error_response(calc_request, step: str, mesaj: str, **kw) -> PriceCalculateResponse:
    """Kısa yardımcı — hata/uyarı response üretir."""
    return PriceCalculateResponse(
        success=False,
        step=step,
        tour_id=calc_request.tour_id,
        tour_name=kw.get("tour_name", ""),
        tarih=kw.get("tarih", calc_request.tur_tarihi),
        kur=kw.get("kur", ""),
        kisibasi_fiyat=kw.get("kisibasi_fiyat", 0.0),
        yetiskin_sayisi=calc_request.yetiskin_sayisi,
        oda_sayisi=kw.get("oda_sayisi", calc_request.oda_sayisi),
        mesaj=mesaj,
        alternatif_tarihler=kw.get("alternatif_tarihler", []),
    )


# =========================================================
# region EMAIL GÖNDERİM FONKSİYONU
# =========================================================

def send_voucher_email(to_email: str, voucher_data: dict) -> bool:
    """
    Rezervasyon voucher bilgilerini müşteriye e-posta olarak gönderir.
    
    Args:
        to_email: Alıcı e-posta adresi
        voucher_data: Voucher bilgileri (dict)
    
    Returns:
        bool: Gönderim başarılı ise True
    """
    try:
        # SMTP ayarları (Yandex)
        smtp_login = os.getenv("SMTP_LOGIN")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_host = os.getenv("SMTP_HOST", "smtp.yandex.com")
        smtp_port = 587

        # E-posta içeriği oluştur
        subject = f"Rezervasyon Onayı - {voucher_data['rezervasyon_no']}"
        
        # HTML formatında voucher
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .voucher {{ border: 2px solid #4CAF50; padding: 20px; max-width: 600px; margin: auto; }}
                .header {{ background-color: #4CAF50; color: white; padding: 15px; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .total {{ font-size: 18px; font-weight: bold; color: #4CAF50; }}
            </style>
        </head>
        <body>
            <div class="voucher">
                <div class="header">
                    <h1>🎫 Rezervasyon Voucher</h1>
                    <h2>{voucher_data['rezervasyon_no']}</h2>
                </div>
                
                <table>
                    <tr><th>🏢 Firma</th><td>{voucher_data['firma']['ad']}</td></tr>
                    <tr><th>👤 Müşteri</th><td>{voucher_data['musteri']['ad_soyad']}</td></tr>
                    <tr><th>📞 Telefon</th><td>{voucher_data['musteri']['telefon']}</td></tr>
                    <tr><th>✉️ E-posta</th><td>{voucher_data['musteri']['email']}</td></tr>
                    <tr><th>🎫 Tur</th><td>[{voucher_data['tur']['kod']}] {voucher_data['tur']['ad']}</td></tr>
                    <tr><th>📅 Başlangıç</th><td>{voucher_data['tur']['baslangic_tarihi']}</td></tr>
                    <tr><th>⏱️ Süre</th><td>{voucher_data['tur']['sure']}</td></tr>
                    <tr><th>🏨 Oda Sayısı</th><td>{voucher_data['konaklama']['oda_sayisi']}</td></tr>
                    <tr><th>👥 Yetişkin</th><td>{voucher_data['konaklama']['yetiskin_sayisi']}</td></tr>
                    <tr><th>👶 Çocuk</th><td>{voucher_data['konaklama']['cocuk_sayisi']}</td></tr>
                    <tr><th class="total">💰 Toplam Fiyat</th>
                        <td class="total">{voucher_data['fiyat']['toplam']:,.0f} {voucher_data['fiyat']['kur']}</td></tr>
                </table>
                
                <p style="margin-top: 20px; text-align: center;">
                    Bizi tercih ettiğiniz için teşekkür ederiz! 🙏<br>
                    İyi tatiller dileriz! 🌴✈️
                </p>
            </div>
        </body>
        </html>
        """
        
        # E-posta oluştur
        msg = MIMEText(html_content, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = smtp_login
        msg['To'] = to_email

        # E-posta gönder (retry ile)
        max_email_retries = 3
        last_error = None
        
        for attempt in range(1, max_email_retries + 1):
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp_server:
                    smtp_server.starttls()
                    smtp_server.login(smtp_login, smtp_password)
                    smtp_server.sendmail(smtp_login, [to_email], msg.as_string())
                    logger.info(f"📧 Voucher e-postası gönderildi: {to_email}")
                    return True
            except smtplib.SMTPConnectError as e:
                last_error = e
                logger.warning(
                    f"[{ErrorCode.SMTP_CONNECTION_ERROR.code}] "
                    f"SMTP bağlantı hatası (deneme {attempt}/{max_email_retries}): {e}"
                )
            except smtplib.SMTPAuthenticationError as e:
                logger.error(
                    f"[{ErrorCode.SMTP_CONNECTION_ERROR.code}] "
                    f"SMTP kimlik doğrulama hatası: {e}"
                )
                return False  # Auth hatası retry yapılmaz
            except smtplib.SMTPException as e:
                last_error = e
                logger.warning(
                    f"[{ErrorCode.EMAIL_SEND_ERROR.code}] "
                    f"SMTP hatası (deneme {attempt}/{max_email_retries}): {e}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{ErrorCode.EMAIL_SEND_ERROR.code}] "
                    f"E-posta hatası (deneme {attempt}/{max_email_retries}): {e}"
                )
            
            if attempt < max_email_retries:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(
            f"[{ErrorCode.EMAIL_SEND_ERROR.code}] "
            f"E-posta gönderilemedi ({max_email_retries} deneme): {last_error}"
        )
        return False
        
    except Exception as e:
        logger.error(
            f"[{ErrorCode.EMAIL_SEND_ERROR.code}] "
            f"E-posta hazırlama hatası: {e}\n{traceback.format_exc()}"
        )
        return False


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
    
    Retry mekanizması dahil (3 deneme, exponential backoff).
    """
    MNG_URL = "https://www.example.com/index.php"
    MNG_AUTH = "Basic bW5nYWl1c2VyOm1uZ2FpcGFzcw=="
    
    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
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
                    if attempt > 1:
                        logger.info(
                            f"✅ MNG API yanıt aldı (deneme {attempt}/{max_retries}, tour_id={tour_id})"
                        )
                    else:
                        logger.info("✅ MNG API yanıt aldı (tour_id=%d)", tour_id)
                    return resp.text
                elif resp.status_code >= 500:
                    # Server hatası - retry yapılabilir
                    last_error = f"HTTP {resp.status_code}"
                    logger.warning(
                        f"[{ErrorCode.MNG_API_RESPONSE_ERROR.code}] "
                        f"MNG API server hatası (deneme {attempt}/{max_retries}): "
                        f"{resp.status_code}"
                    )
                else:
                    # Client hatası - retry yapılmaz
                    logger.error(
                        f"[{ErrorCode.MNG_API_RESPONSE_ERROR.code}] "
                        f"MNG API hatası: {resp.status_code} — {resp.text[:200]}"
                    )
                    return None
                    
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.MNG_API_TIMEOUT.code}] "
                f"MNG API timeout (deneme {attempt}/{max_retries}): {e}"
            )
        except httpx.ConnectError as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.MNG_API_CONNECTION_ERROR.code}] "
                f"MNG API bağlantı hatası (deneme {attempt}/{max_retries}): {e}"
            )
        except httpx.RequestError as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.MNG_API_CONNECTION_ERROR.code}] "
                f"MNG API istek hatası (deneme {attempt}/{max_retries}): {e}"
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.MNG_API_CONNECTION_ERROR.code}] "
                f"MNG API hatası (deneme {attempt}/{max_retries}): {type(e).__name__}: {e}"
            )
        
        if attempt < max_retries:
            delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 saniye
            logger.info(f"⏳ MNG API {delay}s sonra yeniden deneniyor...")
            await asyncio.sleep(delay)
    
    logger.error(
        f"[{ErrorCode.MAX_RETRIES_EXCEEDED.code}] "
        f"MNG API tüm denemeler başarısız ({max_retries} deneme): {last_error}"
    )
    return None


# endregion
# =========================================================


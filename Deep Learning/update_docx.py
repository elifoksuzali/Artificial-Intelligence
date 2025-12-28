# -*- coding: utf-8 -*-
import zipfile
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# Word dosyasını kopyala
shutil.copy('BreakHis_Çalışması.docx', 'BreakHis_Çalışması_backup.docx')
print("Backup oluşturuldu: BreakHis_Çalışması_backup.docx")

# Word dosyasını aç
with zipfile.ZipFile('BreakHis_Çalışması.docx', 'r') as zip_ref:
    zip_ref.extractall('temp_docx')

# document.xml'i oku
doc_path = Path('temp_docx/word/document.xml')
tree = ET.parse(doc_path)
root = tree.getroot()

# Namespace tanımla
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

# Tüm paragrafları bul
paragraphs = root.findall('.//w:p', ns)

print(f"Toplam {len(paragraphs)} paragraf bulundu")

# NAT bölümünü ve eğitim metodolojisini eklemek için metinleri hazırla
nat_text = """Bu çalışmada NAT-Tiny mimarisi kullanılmıştır. Model, 4 aşamalı hiyerarşik encoder yapısına sahiptir. Neighborhood attention pencere boyutu k=7 olarak belirlenmiştir. Patch embedding boyutu 4×4 olup, başlangıç kanal sayısı 64'tür. Her aşamada kanal sayısı iki katına çıkarılmıştır (64 → 128 → 256 → 512). MLP oranı 4 olarak ayarlanmıştır.

NAT-Tiny mimarisinin detaylı yapısı şu şekildedir:
- Aşama 1: 64 boyutlu embedding, 3 NAT bloğu, 2 attention head
- Aşama 2: 128 boyutlu embedding, 4 NAT bloğu, 4 attention head
- Aşama 3: 256 boyutlu embedding, 6 NAT bloğu, 8 attention head
- Aşama 4: 512 boyutlu embedding, 5 NAT bloğu, 16 attention head

Toplam 18 NAT bloğu içeren model, 224×224 piksel giriş görüntülerini 4×4 patch embedding ile 56×56 token dizisine dönüştürür. Her aşamada Patch Merging operatörü ile uzamsal çözünürlük yarıya indirilirken (56×56 → 28×28 → 14×14 → 7×7), embedding boyutu iki katına çıkarılır. Son aşamadan gelen 7×7×512 özellik haritası, global average pooling ile 512 boyutlu bir vektöre indirgenir ve sınıflandırma başlığı üzerinden iki sınıflı (benign/malignant) tahmin üretilir.

Model, NATTEN (Neighborhood Attention Extension) kütüphanesi ile GPU hızlandırması kullanılarak eğitilmiştir. NATTEN, C++ ve CUDA ile optimize edilmiş neighborhood attention çekirdekleri sağlayarak PyTorch fallback implementasyonuna göre yaklaşık 4× daha hızlı çalışma ve %25'e kadar az bellek kullanımı sağlamaktadır."""

print("\nNOT: Word dosyası XML formatında olduğu için doğrudan düzenleme karmaşık.")
print("Lütfen 'NAT_Duzeltme_ve_Egitim_Metodolojisi.txt' dosyasındaki metinleri")
print("Word dosyasına manuel olarak ekleyin.")
print("\n1. Section 3.3 'Method' bölümünü düzeltilmiş metinle değiştirin")
print("2. Section 2'ye '2.4 Eğitim Metodolojisi' bölümünü ekleyin")

# Temizlik
import shutil
shutil.rmtree('temp_docx', ignore_errors=True)

print("\nİşlem tamamlandı!")

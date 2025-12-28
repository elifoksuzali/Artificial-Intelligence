# NAT BreaKHis Model - Detaylı Parametre Dökümanı

## 📋 Genel Bilgiler

**Model Adı:** Neighborhood Attention Transformer (NAT)  
**Veri Seti:** BreaKHis (Breast Cancer Histopathological Image Classification)  
**Sınıf Sayısı:** 2 (Benign, Malignant)  
**Görüntü Boyutu:** 224×224 piksel  
**Toplam Parametre:** ~125,835,010 (125.8M)

---

## 🎯 Model Mimarisi

### NAT Model Yapısı

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Embed Dimension** | 64 | İlk katman embedding boyutu |
| **Depths** | [3, 4, 6, 5] | Her stage'deki blok sayısı |
| **Heads** | [2, 4, 8, 16] | Her stage'deki attention head sayısı |
| **Kernel Size** | 7 | Neighborhood attention kernel boyutu |
| **Patch Size** | 4×4 | Patch embedding boyutu |

### Model Katmanları

1. **Patch Embedding:** 3×224×224 → 64×56×56
2. **Stage 1:** 64 dim, 3 blok, 2 heads → 128×28×28
3. **Stage 2:** 128 dim, 4 blok, 4 heads → 256×14×14
4. **Stage 3:** 256 dim, 6 blok, 8 heads → 512×7×7
5. **Stage 4:** 512 dim, 5 blok, 16 heads → 512×7×7
6. **Classification Head:** 512 → 2 sınıf

---

## ⚙️ Eğitim Parametreleri

### Batch ve Epoch Ayarları

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **BATCH_SIZE** | 256 | Her batch'teki örnek sayısı |
| **EPOCHS** | 50 | Toplam eğitim epoch sayısı |
| **NUM_WORKERS** | 8 | DataLoader worker sayısı |
| **Drop Last** | True | Son eksik batch'i atla |

### Learning Rate ve Optimizer

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **LEARNING_RATE** | 2e-4 (0.0002) | Başlangıç learning rate |
| **MIN_LR** | 1e-6 (0.000001) | Minimum learning rate |
| **WEIGHT_DECAY** | 0.01 | L2 regularization |
| **Optimizer** | AdamW | Adam with Weight Decay |
| **Scheduler** | WarmupCosine | Warmup + Cosine Annealing |

### Learning Rate Schedule

| Faz | Epoch Aralığı | Learning Rate |
|-----|---------------|---------------|
| **Warmup** | 0-9 | 2e-4 × (epoch+1) / 10 |
| **Decay** | 10-44 | Cosine Annealing |
| **Cooldown** | 45-49 | 1e-6 (sabit) |

**Warmup Epochs:** 10  
**Cooldown Epochs:** 5  
**Decay Epochs:** 35

---

## 🎲 Regularization ve Dropout

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **DROP_RATE** | 0.3 | Genel dropout oranı |
| **ATTN_DROP_RATE** | 0.1 | Attention dropout |
| **DROP_PATH_RATE** | 0.1 | Stochastic depth (drop path) |
| **LABEL_SMOOTHING** | 0.1 | Label smoothing faktörü |

### Drop Path Schedule

Drop path rate her blok için linear olarak artar:
- **Stage 1 (3 blok):** 0.0 → 0.0167
- **Stage 2 (4 blok):** 0.0167 → 0.0333
- **Stage 3 (6 blok):** 0.0333 → 0.0667
- **Stage 4 (5 blok):** 0.0667 → 0.1

---

## 🔄 Data Augmentation

### Training Augmentations

| Augmentation | Parametre | Açıklama |
|--------------|-----------|----------|
| **Resize** | 272×272 | Önce büyüt |
| **RandomCrop** | 224×224 | Rastgele kırp |
| **RandomHorizontalFlip** | p=0.5 | Yatay çevir |
| **RandomVerticalFlip** | p=0.5 | Dikey çevir |
| **RandomRotation** | ±30° | Döndür |
| **RandomSharp** | p=0.3, 1.0-2.5x | Keskinleştir |
| **ColorJitter** | 0.3, 0.3, 0.2, 0.1 | Renk değiştir |
| **GaussianBlur** | kernel=3 | Bulanıklaştır |
| **RandomErasing** | p=0.2 | Rastgele sil |

### RandAugment (Opsiyonel)

- **Aktif:** %50 ihtimalle
- **Config:** 'rand-m9-mstd0.5-inc1'
- **Kullanım:** timm kütüphanesi ile

### Mixup & CutMix

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **USE_MIXUP_OR_CUTMIX** | True | Her batch'te birini seç |
| **MIXUP_ALPHA** | 0.2 | Mixup alpha parametresi |
| **CUTMIX_ALPHA** | 1.0 | CutMix alpha parametresi |
| **Seçim Olasılığı** | %50 Mixup, %50 CutMix | Rastgele seçim |

---

## 📊 Veri Seti Ayarları

### Veri Seti İstatistikleri

| Kategori | Değer |
|----------|-------|
| **Toplam Görüntü** | 7,909 |
| **Benign** | 2,480 (%31.4) |
| **Malignant** | 5,429 (%68.6) |
| **Dengesizlik Oranı** | 2.19:1 |

### Veri Bölme (Patient-Level Split)

| Split | Oran | Açıklama |
|-------|------|----------|
| **Train** | 70% | Eğitim için |
| **Validation** | 15% | Doğrulama için |
| **Test** | 15% | Final test için |

**Önemli:** Split patient-level yapılıyor (hasta bazında)

### Sampling Stratejisi

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **USE_UNDERSAMPLING** | True | Aktif |
| **USE_OVERSAMPLING** | False | Kapalı |
| **USE_STAIN_NORMALIZATION** | False | Kapalı (yavaş) |

**Undersampling:** Her sınıftan eşit sayıda örnek alınır (min_count)

### Class Weights

Class weights otomatik hesaplanır:
```
weight = total_samples / (num_classes × class_count)
```

Örnek:
- **Benign weight:** ~1.5 (daha az örnek, daha yüksek ağırlık)
- **Malignant weight:** ~0.75 (daha fazla örnek, daha düşük ağırlık)

---

## 🎯 Loss Function

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Loss Type** | CrossEntropyLoss | Standart cross-entropy |
| **Class Weights** | Otomatik | Dengesiz veri için |
| **Label Smoothing** | 0.1 | Overfitting önleme |

**Loss = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)**

---

## 🔍 Early Stopping

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **PATIENCE** | 15 | 15 epoch iyileşme yoksa dur |
| **Best Metric** | F1-Score | En iyi model seçimi |
| **Monitor** | Validation F1 | Validation F1'i izle |

---

## 🚀 Test-Time Augmentation (TTA)

| Parametre | Değer | Durum |
|-----------|-------|-------|
| **USE_TTA** | True | Aktif |
| **TTA Transforms** | 6 adet | Aşağıda listelenmiş |

### TTA Transform Listesi

1. **Orijinal:** 224×224 resize
2. **Horizontal Flip:** Yatay çevir
3. **Vertical Flip:** Dikey çevir
4. **Rotate +90°:** Saat yönünde 90°
5. **Rotate -90°:** Saat yönü tersi 90°
6. **Larger Scale Crop:** 256×256 → CenterCrop 224

**TTA Sonucu:** 6 tahminin ortalaması alınır

---

## 📈 Metrikler

### Eğitim Sırasında İzlenen Metrikler

| Metrik | Açıklama |
|--------|----------|
| **Train Loss** | Eğitim loss'u |
| **Train Accuracy** | Eğitim doğruluğu |
| **Val Loss** | Validation loss'u |
| **Val Accuracy** | Validation doğruluğu |
| **Val F1-Score** | Validation F1 (best model seçimi) |
| **Val AUC-ROC** | Validation AUC |
| **Learning Rate** | Anlık learning rate |

### Test Metrikleri

| Metrik | Tip | Açıklama |
|--------|-----|----------|
| **Accuracy** | Genel | Toplam doğruluk |
| **Precision** | Weighted/Macro/Per-Class | Hassasiyet |
| **Recall** | Weighted/Macro/Per-Class | Duyarlılık |
| **F1-Score** | Weighted/Macro/Per-Class | Harmonik ortalama |
| **AUC-ROC** | Genel | ROC eğrisi altındaki alan |

### Magnification Bazında Metrikler

Her magnification seviyesi için ayrı metrikler:
- **40X, 100X, 200X, 400X** için ayrı ayrı

---

## 💾 Model Kaydetme

### Checkpoint Dosyaları

| Dosya | Açıklama |
|-------|----------|
| **nat_best.pth** | En iyi validation F1 modeli |
| **nat_best_40X.pth** | 40X için en iyi model |
| **nat_best_100X.pth** | 100X için en iyi model |
| **nat_best_200X.pth** | 200X için en iyi model |
| **nat_best_400X.pth** | 400X için en iyi model |

---

## 🎲 Seed ve Reproducibility

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **BASE_SEED** | 42 | Temel seed |
| **ENSEMBLE_SEEDS** | [42, 123, 456, 789] | Ensemble için 4 farklı seed |
| **CURRENT_SEED** | 42 | Şu anki seed |

**Not:** Ensemble training için her model farklı seed kullanır

---

## 🔧 NATTEN Optimizasyonu

| Parametre | Değer | Durum |
|-----------|-------|-------|
| **USE_NATTEN** | Otomatik | Yüklüyse True |
| **NATTEN_API_STYLE** | 'new' | Yeni API (0.21.x) |
| **Fallback** | PyTorch | NATTEN yoksa |

**NATTEN:** GPU-optimized neighborhood attention

---

## 📥 Pretrained Model

| Parametre | Değer | Durum |
|-----------|-------|-------|
| **USE_PRETRAINED** | True | Aktif |
| **Source** | ImageNet | Pretrained ağırlıklar |
| **URL 1** | shi-labs.com/projects/nat/... | Birincil kaynak |
| **URL 2** | huggingface.co/shi-labs/... | Yedek kaynak |

**Not:** Classification head hariç tüm katmanlar yüklenir

---

## 🖼️ Görüntü Normalizasyonu

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Mean** | [0.485, 0.456, 0.406] | ImageNet mean |
| **Std** | [0.229, 0.224, 0.225] | ImageNet std |

---

## 📊 Çıktı Dosyaları

| Dosya | Açıklama |
|-------|----------|
| **results.png** | Eğitim grafikleri ve confusion matrix |
| **nat_best.pth** | En iyi model checkpoint |
| **nat_best_*.pth** | Magnification bazında checkpoint'ler |

---

## 🎯 Beklenen Performans

### Önceki Sonuçlar (Referans)

| Metrik | Değer |
|--------|-------|
| **Test Accuracy** | 82.35% |
| **Test F1-Score** | 82.48% |
| **Test Precision** | 82.69% |
| **Test Recall** | 82.35% |
| **Benign F1** | 72.40% |
| **Malignant F1** | 87.02% |

### Pretrained ile Beklenen İyileşme

| Metrik | Önce | Sonra (Beklenen) |
|--------|------|------------------|
| **Test Accuracy** | 82.35% | **85-88%** |
| **Test F1-Score** | 82.48% | **85-88%** |
| **Epoch Sayısı** | 50 | **20-30** |

---

## 📝 Önemli Notlar

1. **Patient-Level Split:** Veri seti hasta bazında bölünür (data leakage önleme)
2. **Class Weights:** Dengesiz veri için otomatik hesaplanır
3. **Mixed Precision:** AMP (Automatic Mixed Precision) aktif
4. **TTA:** Test sırasında 6 farklı transform uygulanır
5. **Ensemble Ready:** 4 farklı seed ile ensemble yapılabilir
6. **Pretrained:** ImageNet ağırlıkları ile başlar (daha hızlı öğrenme)

---

## 🔄 Kod Yapısı

### Ana Bölümler

1. **Imports & Setup** (Satır 1-80)
2. **Dataset Download** (Satır 82-89)
3. **Config** (Satır 91-157)
4. **Stain Normalization** (Satır 159-236)
5. **NAT Model** (Satır 238-381)
6. **Data Preparation** (Satır 383-458)
7. **Transforms & Dataset** (Satır 460-530)
8. **Training Setup** (Satır 532-652)
9. **Mixup & CutMix** (Satır 654-722)
10. **Training Loop** (Satır 724-851)
11. **Evaluation & TTA** (Satır 853-1001)
12. **Visualization** (Satır 1003-1027)

---

**Son Güncelleme:** Kod analizi tarihi  
**Versiyon:** colab_nat_v2_3.py

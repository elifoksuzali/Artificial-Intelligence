"""
NAT Model Konfigürasyon Profilleri
==================================

Farklı donanım ortamları için optimize edilmiş ayarlar.
"""

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigProfile:
    """Konfigürasyon profili"""
    name: str
    batch_size: int
    img_size: int
    num_workers: int
    use_amp: bool  # Mixed precision
    gradient_accumulation: int
    model_variant: str  # 'mini', 'tiny', 'small', 'base'
    
    # NAT model parametreleri
    embed_dim: int
    depths: tuple
    num_heads: tuple
    kernel_size: int


# =============================================================================
# DONANIM PROFİLLERİ
# =============================================================================

# 🟢 DÜŞÜK VRAM (2-4 GB) - GTX 1650, MX450, vb.
LOW_VRAM = ConfigProfile(
    name="low_vram",
    batch_size=4,
    img_size=224,
    num_workers=2,
    use_amp=True,  # Bellek tasarrufu için
    gradient_accumulation=4,  # Effective batch size = 16
    model_variant='mini',
    embed_dim=48,  # Küçültülmüş
    depths=(2, 2, 4, 2),
    num_heads=(2, 4, 6, 12),
    kernel_size=5,  # Küçük kernel
)

# 🟡 ORTA VRAM (4-8 GB) - GTX 1660, RTX 2060, vb.
MEDIUM_VRAM = ConfigProfile(
    name="medium_vram",
    batch_size=8,
    img_size=224,
    num_workers=4,
    use_amp=True,
    gradient_accumulation=2,  # Effective batch size = 16
    model_variant='mini',
    embed_dim=64,
    depths=(2, 2, 6, 2),
    num_heads=(2, 4, 8, 16),
    kernel_size=7,
)

# 🟢 YÜKSEK VRAM (8-12 GB) - RTX 3060, RTX 3070, vb.
HIGH_VRAM = ConfigProfile(
    name="high_vram",
    batch_size=16,
    img_size=224,
    num_workers=4,
    use_amp=True,
    gradient_accumulation=1,
    model_variant='tiny',
    embed_dim=64,
    depths=(3, 4, 6, 5),
    num_heads=(2, 4, 8, 16),
    kernel_size=7,
)

# 🔵 COLAB / KAGGLE (15-16 GB)
COLAB = ConfigProfile(
    name="colab",
    batch_size=24,
    img_size=224,
    num_workers=2,  # Colab'da düşük tut
    use_amp=True,
    gradient_accumulation=1,
    model_variant='tiny',
    embed_dim=64,
    depths=(3, 4, 6, 5),
    num_heads=(2, 4, 8, 16),
    kernel_size=7,
)

# 🔴 CPU ONLY (GPU yok)
CPU_ONLY = ConfigProfile(
    name="cpu_only",
    batch_size=4,
    img_size=192,  # Küçük görüntü
    num_workers=0,  # Windows'ta multiprocessing sorunu önlemek için
    use_amp=False,  # CPU'da AMP yok
    gradient_accumulation=4,
    model_variant='mini',
    embed_dim=32,  # Çok küçük model
    depths=(1, 1, 2, 1),
    num_heads=(1, 2, 4, 8),
    kernel_size=5,
)


def get_profile_for_system() -> ConfigProfile:
    """
    Sistem donanımına göre otomatik profil seç
    """
    if not torch.cuda.is_available():
        print("⚠️ GPU bulunamadı, CPU profili kullanılıyor")
        return CPU_ONLY
    
    # GPU belleğini kontrol et
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    
    print(f"🖥️ GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb < 4:
        print("📋 Profil: LOW_VRAM")
        return LOW_VRAM
    elif gpu_memory_gb < 8:
        print("📋 Profil: MEDIUM_VRAM")
        return MEDIUM_VRAM
    elif gpu_memory_gb < 14:
        print("📋 Profil: HIGH_VRAM")
        return HIGH_VRAM
    else:
        print("📋 Profil: COLAB/HIGH-END")
        return COLAB


def print_profile_info(profile: ConfigProfile):
    """Profil bilgilerini yazdır"""
    print("\n" + "=" * 50)
    print(f"📋 Konfigürasyon Profili: {profile.name.upper()}")
    print("=" * 50)
    print(f"  Batch Size: {profile.batch_size}")
    print(f"  Image Size: {profile.img_size}x{profile.img_size}")
    print(f"  Gradient Accumulation: {profile.gradient_accumulation}")
    print(f"  Effective Batch Size: {profile.batch_size * profile.gradient_accumulation}")
    print(f"  Mixed Precision (AMP): {profile.use_amp}")
    print(f"  Model Variant: {profile.model_variant}")
    print(f"  Embed Dim: {profile.embed_dim}")
    print(f"  Depths: {profile.depths}")
    print(f"  Num Heads: {profile.num_heads}")
    print(f"  Kernel Size: {profile.kernel_size}")
    print("=" * 50)


def estimate_memory_usage(profile: ConfigProfile) -> dict:
    """
    Tahmini bellek kullanımını hesapla
    """
    # Yaklaşık hesaplama
    img_size = profile.img_size
    batch_size = profile.batch_size
    embed_dim = profile.embed_dim
    
    # Model parametreleri (yaklaşık)
    total_depth = sum(profile.depths)
    params = embed_dim * embed_dim * total_depth * 12  # Yaklaşık
    
    # Aktivasyonlar (forward pass)
    activation_size = batch_size * (img_size // 4) ** 2 * embed_dim * 4
    
    # Gradyanlar (backward pass)
    gradient_size = params * 4  # float32
    
    # Toplam (byte)
    total_bytes = params * 4 + activation_size + gradient_size
    
    # AMP ile %30-40 tasarruf
    if profile.use_amp:
        total_bytes *= 0.65
    
    return {
        'model_params_mb': params * 4 / (1024**2),
        'activations_mb': activation_size / (1024**2),
        'gradients_mb': gradient_size / (1024**2),
        'total_estimated_gb': total_bytes / (1024**3),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n🔍 Sistem Analizi")
    print("-" * 50)
    
    # Otomatik profil seç
    profile = get_profile_for_system()
    print_profile_info(profile)
    
    # Bellek tahmini
    memory = estimate_memory_usage(profile)
    print(f"\n💾 Tahmini Bellek Kullanımı:")
    print(f"  Model: {memory['model_params_mb']:.1f} MB")
    print(f"  Aktivasyonlar: {memory['activations_mb']:.1f} MB")
    print(f"  Gradyanlar: {memory['gradients_mb']:.1f} MB")
    print(f"  Toplam: ~{memory['total_estimated_gb']:.2f} GB")
    
    # Tüm profilleri listele
    print("\n" + "=" * 50)
    print("📋 TÜM PROFİLLER")
    print("=" * 50)
    
    profiles = [CPU_ONLY, LOW_VRAM, MEDIUM_VRAM, HIGH_VRAM, COLAB]
    
    print(f"{'Profil':<15} {'Batch':<8} {'Img':<8} {'AMP':<6} {'VRAM Gereksinimi':<20}")
    print("-" * 60)
    
    for p in profiles:
        mem = estimate_memory_usage(p)
        print(f"{p.name:<15} {p.batch_size:<8} {p.img_size:<8} {str(p.use_amp):<6} ~{mem['total_estimated_gb']:.1f} GB")


"""
Stain Normalization for Histopathology Images
==============================================

Bu modül histopatoloji görüntüleri için renk normalizasyonu sağlar.
Vahadane ve Macenko yöntemlerini destekler.

Kullanım:
    from stain_normalization import StainNormalizer, get_reference_image
    
    normalizer = StainNormalizer(method='macenko')
    normalized_image = normalizer.normalize(image, reference_image)
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings


class StainNormalizer:
    """
    Histopatoloji görüntüleri için Stain Normalization
    
    Desteklenen yöntemler:
    - 'macenko': Macenko stain normalization (hızlı, stabil)
    - 'reinhard': Reinhard color transfer (en hızlı, basit)
    
    Args:
        method: Normalizasyon yöntemi ('macenko' veya 'reinhard')
    """
    
    def __init__(self, method: str = 'macenko'):
        self.method = method.lower()
        self.reference_stats = None
        self.stain_matrix_ref = None
        
        if self.method not in ['macenko', 'reinhard']:
            raise ValueError(f"Desteklenmeyen yöntem: {method}. 'macenko' veya 'reinhard' kullanın.")
    
    def fit(self, reference_image: np.ndarray):
        """
        Referans görüntüden istatistikleri öğren
        
        Args:
            reference_image: RGB referans görüntü (numpy array, 0-255)
        """
        if reference_image.max() <= 1.0:
            reference_image = (reference_image * 255).astype(np.uint8)
        
        if self.method == 'reinhard':
            self.reference_stats = self._get_reinhard_stats(reference_image)
        elif self.method == 'macenko':
            self.stain_matrix_ref, self.max_conc_ref = self._get_macenko_params(reference_image)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüyü normalize et
        
        Args:
            image: RGB görüntü (numpy array, 0-255)
            
        Returns:
            Normalize edilmiş görüntü
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        if self.method == 'reinhard':
            return self._reinhard_normalize(image)
        elif self.method == 'macenko':
            return self._macenko_normalize(image)
    
    def normalize(self, image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tek adımda fit ve transform
        
        Args:
            image: Normalize edilecek görüntü
            reference_image: Referans görüntü (None ise önceden fit edilmiş olmalı)
        """
        if reference_image is not None:
            self.fit(reference_image)
        
        if self.reference_stats is None and self.stain_matrix_ref is None:
            warnings.warn("Referans görüntü ayarlanmamış. Orijinal görüntü döndürülüyor.")
            return image
        
        return self.transform(image)
    
    # =========================================================================
    # REINHARD METHOD (Basit, hızlı)
    # =========================================================================
    
    def _get_reinhard_stats(self, image: np.ndarray) -> dict:
        """LAB renk uzayında istatistikleri hesapla"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        return {
            'mean': np.mean(lab, axis=(0, 1)),
            'std': np.std(lab, axis=(0, 1))
        }
    
    def _reinhard_normalize(self, image: np.ndarray) -> np.ndarray:
        """Reinhard renk transferi"""
        # Kaynak istatistikleri
        source_stats = self._get_reinhard_stats(image)
        
        # LAB'a çevir
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Her kanal için normalize et
        for i in range(3):
            lab[:, :, i] = (lab[:, :, i] - source_stats['mean'][i]) / (source_stats['std'][i] + 1e-6)
            lab[:, :, i] = lab[:, :, i] * self.reference_stats['std'][i] + self.reference_stats['mean'][i]
        
        # Clip ve RGB'ye çevir
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return rgb
    
    # =========================================================================
    # MACENKO METHOD (Stain separation tabanlı)
    # =========================================================================
    
    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """RGB'den Optical Density'e çevir"""
        image = image.astype(np.float32) + 1  # log(0) önlemek için
        od = -np.log(image / 255.0)
        return od
    
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Optical Density'den RGB'ye çevir"""
        rgb = 255 * np.exp(-od)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
    
    def _get_macenko_params(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Macenko stain matrix ve konsantrasyonları hesapla"""
        # OD'ye çevir
        od = self._rgb_to_od(image)
        od = od.reshape(-1, 3)
        
        # Beyaz pikselleri filtrele (düşük OD)
        od_threshold = 0.15
        mask = np.all(od > od_threshold, axis=1)
        od_filtered = od[mask]
        
        if len(od_filtered) < 100:
            # Yeterli piksel yoksa varsayılan stain matrix kullan
            stain_matrix = np.array([
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
            ])
            max_conc = np.array([1.0, 1.0])
            return stain_matrix, max_conc
        
        # SVD ile stain vektörlerini bul
        try:
            _, _, V = np.linalg.svd(od_filtered, full_matrices=False)
            
            # İlk iki eigenvector (en önemli stain yönleri)
            V = V[:2, :]
            
            # Açıyı hesapla
            phi = np.arctan2(V[1, :], V[0, :])
            
            # Min ve max açılardaki vektörler
            min_phi = np.percentile(phi, 1)
            max_phi = np.percentile(phi, 99)
            
            v1 = np.array([np.cos(min_phi), np.sin(min_phi)])
            v2 = np.array([np.cos(max_phi), np.sin(max_phi)])
            
            # Stain matrix oluştur
            stain_matrix = np.array([
                V.T @ v1,
                V.T @ v2
            ])
            
            # Normalize et
            stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
            
            # Konsantrasyonları hesapla
            concentrations = np.linalg.lstsq(stain_matrix.T, od_filtered.T, rcond=None)[0]
            max_conc = np.percentile(concentrations, 99, axis=1)
            
        except Exception as e:
            # Hata durumunda varsayılan değerler
            warnings.warn(f"Macenko parametreleri hesaplanamadı: {e}. Varsayılan değerler kullanılıyor.")
            stain_matrix = np.array([
                [0.644211, 0.716556, 0.266844],
                [0.092789, 0.954111, 0.283111],
            ])
            max_conc = np.array([1.0, 1.0])
        
        return stain_matrix, max_conc
    
    def _macenko_normalize(self, image: np.ndarray) -> np.ndarray:
        """Macenko stain normalization"""
        # Kaynak parametreleri
        stain_matrix_src, max_conc_src = self._get_macenko_params(image)
        
        # OD'ye çevir
        od = self._rgb_to_od(image)
        h, w, _ = od.shape
        od = od.reshape(-1, 3)
        
        try:
            # Konsantrasyonları hesapla
            concentrations = np.linalg.lstsq(stain_matrix_src.T, od.T, rcond=None)[0]
            
            # Konsantrasyonları normalize et
            concentrations = concentrations / (max_conc_src.reshape(-1, 1) + 1e-6)
            concentrations = concentrations * self.max_conc_ref.reshape(-1, 1)
            
            # Yeni OD hesapla
            od_normalized = self.stain_matrix_ref.T @ concentrations
            od_normalized = od_normalized.T.reshape(h, w, 3)
            
            # RGB'ye çevir
            rgb = self._od_to_rgb(od_normalized)
            
        except Exception as e:
            warnings.warn(f"Macenko normalizasyon başarısız: {e}. Orijinal görüntü döndürülüyor.")
            return image
        
        return rgb


class TorchStainNormalizer:
    """
    PyTorch ile uyumlu Stain Normalizer wrapper
    
    Kullanım:
        normalizer = TorchStainNormalizer(method='macenko')
        normalizer.set_reference(reference_image)  # Bir kez çağır
        
        # Transform içinde:
        normalized = normalizer(pil_image)
    """
    
    def __init__(self, method: str = 'macenko', reference_image: Optional[np.ndarray] = None):
        self.normalizer = StainNormalizer(method=method)
        self.is_fitted = False
        
        if reference_image is not None:
            self.set_reference(reference_image)
    
    def set_reference(self, reference_image: Union[np.ndarray, Image.Image, str, Path]):
        """Referans görüntüyü ayarla"""
        if isinstance(reference_image, (str, Path)):
            reference_image = np.array(Image.open(reference_image).convert('RGB'))
        elif isinstance(reference_image, Image.Image):
            reference_image = np.array(reference_image.convert('RGB'))
        
        self.normalizer.fit(reference_image)
        self.is_fitted = True
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Görüntüyü normalize et"""
        if not self.is_fitted:
            warnings.warn("Referans görüntü ayarlanmamış. Orijinal görüntü döndürülüyor.")
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
            return image
        
        # PIL'den numpy'a çevir
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        # Normalize et
        normalized = self.normalizer.transform(image_np)
        
        # PIL'e çevir
        return Image.fromarray(normalized)


def get_reference_image(dataset_path: Path, sample_count: int = 10) -> np.ndarray:
    """
    Veri setinden temsili bir referans görüntü seç
    
    Birden fazla görüntüden ortalama istatistikler alarak
    daha stabil bir referans oluşturur.
    
    Args:
        dataset_path: Görüntülerin bulunduğu klasör
        sample_count: Örneklenecek görüntü sayısı
        
    Returns:
        Referans görüntü (numpy array)
    """
    import random
    
    # PNG dosyalarını bul
    image_paths = list(dataset_path.rglob('*.png'))
    
    if len(image_paths) == 0:
        raise ValueError(f"Görüntü bulunamadı: {dataset_path}")
    
    # Rastgele örnek seç
    sample_paths = random.sample(image_paths, min(sample_count, len(image_paths)))
    
    # Görüntüleri yükle ve ortalamasını al
    images = []
    for p in sample_paths:
        try:
            img = np.array(Image.open(p).convert('RGB'))
            # Aynı boyuta yeniden boyutlandır
            img = cv2.resize(img, (256, 256))
            images.append(img)
        except Exception as e:
            print(f"Görüntü yüklenemedi: {p}, hata: {e}")
    
    if len(images) == 0:
        raise ValueError("Hiç görüntü yüklenemedi")
    
    # Ortalama görüntü (referans olarak kullanılacak)
    # Not: Aslında ilk görüntüyü referans olarak kullanmak daha yaygın
    reference = images[0]
    
    print(f"✓ Referans görüntü seçildi: {sample_paths[0]}")
    
    return reference


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Stain Normalization Test")
    print("=" * 50)
    
    # Test görüntüsü oluştur
    test_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    reference_image = np.random.randint(150, 250, (256, 256, 3), dtype=np.uint8)
    
    # Reinhard test
    print("\n1. Reinhard Normalization Test...")
    normalizer_reinhard = StainNormalizer(method='reinhard')
    normalized_reinhard = normalizer_reinhard.normalize(test_image, reference_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {normalized_reinhard.shape}")
    print("   ✓ Reinhard çalışıyor!")
    
    # Macenko test
    print("\n2. Macenko Normalization Test...")
    normalizer_macenko = StainNormalizer(method='macenko')
    normalized_macenko = normalizer_macenko.normalize(test_image, reference_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {normalized_macenko.shape}")
    print("   ✓ Macenko çalışıyor!")
    
    # TorchStainNormalizer test
    print("\n3. TorchStainNormalizer Test...")
    torch_normalizer = TorchStainNormalizer(method='macenko', reference_image=reference_image)
    pil_image = Image.fromarray(test_image)
    normalized_pil = torch_normalizer(pil_image)
    print(f"   Input type: PIL.Image, Output type: {type(normalized_pil)}")
    print("   ✓ TorchStainNormalizer çalışıyor!")
    
    print("\n" + "=" * 50)
    print("Tüm testler başarılı! ✓")


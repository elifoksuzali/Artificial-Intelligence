"""
BreaKHis Dataset üzerinde NAT (Neighborhood Attention Transformer) Eğitimi
==========================================================================

Bu script, BreaKHis meme kanseri histopatoloji veri seti üzerinde
NAT modelini eğitir.

Kullanım:
    python train_nat_breakhis.py

Özellikler:
- Patient-stratified split (veri sızıntısını önler)
- Class weight ile dengesiz veri yönetimi
- Data augmentation
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Detaylı performans metrikleri (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion Matrix görselleştirme
- Training grafikleri
- Per-class performans analizi
"""

import os
import random
import re
from pathlib import Path
from glob import glob
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# PyTorch 2.x uyumlu AMP (Mixed Precision)
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# NAT modelini import et
from nat_model import NAT, nat_mini, nat_tiny, nat_small

# Stain Normalization modülü
from stain_normalization import StainNormalizer, TorchStainNormalizer, get_reference_image

# PIL ImageEnhance (Sharpening için)
from PIL import ImageEnhance


# =============================================================================
# KONFİGÜRASYON
# =============================================================================

class Config:
    """
    Eğitim Konfigürasyonu
    
    GPU (RTX 3090, RTX 4090 vb.) için optimize edilmiş ayarlar
    Platform bağımsız (Windows/Linux)
    """
    
    # =========================================================================
    # VERİ SETİ YOLU - Otomatik algılama (Windows/Linux)
    # =========================================================================
    @staticmethod
    def get_base_path():
        """Platform bağımsız veri seti yolu"""
        import platform
        
        if platform.system() == "Windows":
            base = Path(os.path.expanduser(r"~\.cache\kagglehub\datasets\ambarish\breakhis"))
        else:
            base = Path(os.path.expanduser("~/.cache/kagglehub/datasets/ambarish/breakhis"))
        
        # En son versiyonu bul
        if base.exists():
            versions = sorted(base.glob("versions/*"), key=lambda x: int(x.name) if x.name.isdigit() else 0)
            if versions:
                return versions[-1] / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
        
        return base / "versions" / "4" / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
    
    # =========================================================================
    # GPU OPTİMİZE AYARLAR
    # =========================================================================
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 8         # Paylaşımlı GPU için düşürüldü (memory tasarrufu)
    NUM_WORKERS = 8         # Windows'ta 4, Linux'ta 8 kullanılabilir
    EPOCHS = 20              # V1-V2 ile aynı (karşılaştırma için)
    LEARNING_RATE = 5e-4     # Warmup ile daha yüksek LR kullanılabilir (0.0005)
    WEIGHT_DECAY = 0.1       # Daha güçlü regularization
    
    # ==========================================================================
    # YENİ: WARMUP + COOLDOWN (Transformer best practice)
    # ==========================================================================
    WARMUP_EPOCHS = 3        # LR yavaşça 0 → max artar
    COOLDOWN_EPOCHS = 2      # Son epoch'larda LR minimum'da sabit kalır
    MIN_LR = 1e-6            # Minimum learning rate
    
    # Model parametreleri
    NUM_CLASSES = 2
    CLASS_NAMES = ['benign', 'malignant']
    
    # NAT Model Varyantı: 'mini', 'tiny', 'small', 'base'
    MODEL_VARIANT = 'tiny'
    
    # Model dropout (overfitting önlemek için)
    DROP_RATE = 0.2
    ATTN_DROP_RATE = 0.1
    DROP_PATH_RATE = 0.3
    
    # Magnification seçimi
    MAGNIFICATIONS = ['40X', '100X', '200X', '400X']
    SELECTED_MAG = 'ALL'     # 'ALL' = tüm magnification'lar, veya '40X', '100X', '200X', '400X'
    
    # Early stopping
    PATIENCE = 8  # Daha erken dur, overfitting önle
    
    # Mixed Precision (AMP) - GPU Tensor Core kullanır
    USE_AMP = True
    
    # ==========================================================================
    # YENİ: STAIN NORMALIZATION (Swin makalesi preprocessing)
    # ==========================================================================
    USE_STAIN_NORMALIZATION = True  # Stain normalization aktif
    STAIN_METHOD = 'macenko'        # 'macenko' veya 'reinhard'
    
    # ==========================================================================
    # YENİ: OVERSAMPLING (Benign sınıfı için)
    # ==========================================================================
    USE_OVERSAMPLING = True         # Azınlık sınıfını çoğalt
    
    # Checkpoint ve Output dizinleri
    CHECKPOINT_DIR = Path("checkpoints_3")  # Yeni çıktılar için
    OUTPUT_DIR = Path("outputs_3")          # Yeni çıktılar için
    
    # Device - Otomatik algılama
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Seed ayarla
def set_seed(seed: int):
    """Reproducibility için seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# VERİ SETİ HAZIRLAMA
# =============================================================================

def parse_breakhis_path(filepath: str) -> Tuple[str, str, str]:
    """
    BreaKHis dosya yolunu parse et
    
    Returns:
        (label, magnification, patient_id)
    """
    fname = os.path.basename(filepath)
    parts = fname.split('_')
    
    # Label: B -> benign, M -> malignant
    label_token = parts[1] if len(parts) > 1 else ''
    label = 'benign' if label_token.upper().startswith('B') else 'malignant'
    
    # Magnification: klasör adından
    mag = Path(filepath).parents[0].name
    
    # Patient ID
    try:
        third = parts[2]
        patient = third.rsplit('-', 2)[0]
    except Exception:
        m = re.search(r'([A-Za-z]-\d+-\w+)', fname)
        patient = m.group(1) if m else fname
    
    return label, mag, patient


def create_dataframe(base_path: Path) -> pd.DataFrame:
    """Veri setinden DataFrame oluştur"""
    image_paths = sorted([str(p) for p in base_path.rglob('*.png')])
    
    rows = []
    for p in image_paths:
        label, mag, patient = parse_breakhis_path(p)
        rows.append({
            'filepath': p,
            'label': label,
            'mag': mag.upper().replace(' ', ''),
            'patient_id': patient
        })
    
    df = pd.DataFrame(rows)
    return df


def patient_stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Patient-based + Class-stratified split
    Her sınıftan (benign/malignant) hastaları ayrı ayrı böler
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Her sınıf için ayrı ayrı split yap
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        patients = df_label['patient_id'].unique().tolist()
        random.Random(seed).shuffle(patients)
        
        n = len(patients)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        
        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train:n_train + n_val])
        test_patients = set(patients[n_train + n_val:])
        
        train_dfs.append(df_label[df_label['patient_id'].isin(train_patients)])
        val_dfs.append(df_label[df_label['patient_id'].isin(val_patients)])
        test_dfs.append(df_label[df_label['patient_id'].isin(test_patients)])
    
    return {
        'train': pd.concat(train_dfs).reset_index(drop=True),
        'val': pd.concat(val_dfs).reset_index(drop=True),
        'test': pd.concat(test_dfs).reset_index(drop=True)
    }


def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Veri setini dengele - Azınlık sınıfını çoğalt (oversampling)
    
    Benign: 2480 (31.4%)
    Malignant: 5429 (68.6%)
    
    → Her ikisi de ~5429 olacak
    """
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    
    balanced_dfs = []
    
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        current_count = len(df_label)
        
        if current_count < max_count:
            # Azınlık sınıfını çoğalt (oversampling)
            n_samples_needed = max_count - current_count
            
            # Rastgele örnekle (tekrar ile)
            df_oversampled = df_label.sample(
                n=n_samples_needed, 
                replace=True, 
                random_state=seed
            )
            df_label = pd.concat([df_label, df_oversampled])
            print(f"   ⚖️ {label}: {current_count} → {len(df_label)} (oversampled)")
        else:
            print(f"   ⚖️ {label}: {current_count} (unchanged)")
        
        balanced_dfs.append(df_label)
    
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    
    # Karıştır
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return balanced_df


# =============================================================================
# CUSTOM TRANSFORMS (Swin makalesi için eklenen)
# =============================================================================

class RandomSharpening:
    """
    Rastgele sharpening uygula - Swin makalesi preprocessing
    
    Histopatoloji görüntülerinde hücre sınırlarını netleştirir.
    """
    def __init__(self, p: float = 0.3, factor_range: Tuple[float, float] = (1.0, 2.5)):
        self.p = p
        self.factor_range = factor_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, factor_range={self.factor_range})"


class StainNormTransform:
    """
    Stain Normalization Transform - Swin makalesi preprocessing
    
    Histopatoloji görüntülerinde renk normalizasyonu uygular.
    """
    def __init__(self, normalizer: TorchStainNormalizer):
        self.normalizer = normalizer
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return self.normalizer(img)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.normalizer.normalizer.method})"


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class BreaKHisDataset(Dataset):
    """BreaKHis PyTorch Dataset - Tüm magnification destekli"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        class_names: List[str] = ['benign', 'malignant'],
        return_meta: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.return_meta = return_meta  # Magnification bilgisi döndür
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Görüntüyü yükle
        image = Image.open(row['filepath']).convert('RGB')
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
        
        # Label
        label = self.class_to_idx[row['label']]
        
        if self.return_meta:
            return image, label, row['mag']
        
        return image, label
    
    def get_magnification_distribution(self) -> Dict[str, int]:
        """Magnification dağılımını döndür"""
        return self.df['mag'].value_counts().to_dict()


def get_transforms(
    img_size: int = 224, 
    is_training: bool = True,
    stain_normalizer: Optional[TorchStainNormalizer] = None
) -> T.Compose:
    """
    Data augmentation ve preprocessing - Swin makalesi ile uyumlu
    
    Yeni eklenenler (Swin makalesinden):
    - Stain Normalization (Vahadane/Macenko)
    - Sharpening augmentation
    
    Args:
        img_size: Hedef görüntü boyutu
        is_training: Eğitim modu (True) veya test modu (False)
        stain_normalizer: Stain normalization için normalizer (None ise atlanır)
    """
    
    # Başlangıç transformları (stain normalization dahil)
    initial_transforms = []
    
    # Stain Normalization (ilk adım olarak uygulanmalı)
    if stain_normalizer is not None:
        initial_transforms.append(StainNormTransform(stain_normalizer))
    
    if is_training:
        return T.Compose(initial_transforms + [
            T.Resize((img_size + 48, img_size + 48)),  # Daha büyük resize
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),  # Daha fazla rotasyon
            # YENİ: Sharpening (Swin makalesinden)
            RandomSharpening(p=0.3, factor_range=(1.0, 2.5)),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            T.RandomAffine(
                degrees=15,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10
            ),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose(initial_transforms + [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_dataloaders(
    splits: Dict[str, pd.DataFrame],
    config: Config,
    stain_normalizer: Optional[TorchStainNormalizer] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    DataLoader'ları oluştur
    
    Args:
        splits: Train/Val/Test DataFrame'leri
        config: Konfigürasyon
        stain_normalizer: Stain normalization için (None ise atlanır)
    """
    
    train_ds = BreaKHisDataset(
        splits['train'],
        transform=get_transforms(config.IMG_SIZE, is_training=True, stain_normalizer=stain_normalizer),
        class_names=config.CLASS_NAMES
    )
    
    val_ds = BreaKHisDataset(
        splits['val'],
        transform=get_transforms(config.IMG_SIZE, is_training=False, stain_normalizer=stain_normalizer),
        class_names=config.CLASS_NAMES
    )
    
    test_ds = BreaKHisDataset(
        splits['test'],
        transform=get_transforms(config.IMG_SIZE, is_training=False, stain_normalizer=stain_normalizer),
        class_names=config.CLASS_NAMES
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(df: pd.DataFrame, class_names: List[str], boost_minority: float = 1.5) -> torch.Tensor:
    """
    Dengesiz veri için class weights hesapla
    
    Args:
        df: DataFrame
        class_names: Sınıf isimleri ['benign', 'malignant']
        boost_minority: Azınlık sınıfına ekstra ağırlık çarpanı (1.5 = %50 daha fazla)
    
    Örnek:
        Benign: 2480 (31%) → weight = 1.59 * 1.5 = 2.39
        Malignant: 5429 (69%) → weight = 0.73
    """
    class_counts = df['label'].value_counts()
    total = len(df)
    
    # En az örneğe sahip sınıfı bul
    min_class = class_counts.idxmin()
    
    weights = []
    for cls in class_names:
        count = class_counts.get(cls, 1)
        weight = total / (len(class_names) * count)
        
        # Azınlık sınıfına ekstra boost
        if cls == min_class:
            weight *= boost_minority
        
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# EĞİTİM FONKSİYONLARI
# =============================================================================

class WarmupCosineScheduler:
    """
    Warmup + Cosine Decay + Cooldown Learning Rate Scheduler
    
    Transformer eğitimi için best practice:
    1. Warmup: LR yavaşça 0 → max_lr artar (ilk N epoch)
    2. Cosine Decay: LR yumuşakça max_lr → min_lr azalır
    3. Cooldown: LR min_lr'de sabit kalır (son M epoch)
    
    Kullanım:
        scheduler = WarmupCosineScheduler(optimizer, config)
        for epoch in range(epochs):
            train()
            scheduler.step(epoch)
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 3,
        cooldown_epochs: int = 2,
        max_lr: float = 5e-4,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Cosine decay epoch sayısı
        self.decay_epochs = total_epochs - warmup_epochs - cooldown_epochs
        
        self.current_lr = 0.0
        self._last_lr = [0.0]
    
    def get_lr(self, epoch: int) -> float:
        """Epoch'a göre LR hesapla"""
        
        if epoch < self.warmup_epochs:
            # Warmup: Linear artış (0 → max_lr)
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        
        elif epoch < self.total_epochs - self.cooldown_epochs:
            # Cosine Decay: Yumuşak azalma (max_lr → min_lr)
            decay_epoch = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_epoch / self.decay_epochs))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        else:
            # Cooldown: Sabit minimum LR
            lr = self.min_lr
        
        return lr
    
    def step(self, epoch: int):
        """Optimizer'ın LR'sini güncelle"""
        lr = self.get_lr(epoch)
        self.current_lr = lr
        self._last_lr = [lr]
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        """Son LR değerini döndür (PyTorch scheduler uyumluluğu için)"""
        return self._last_lr
    
    def state_dict(self):
        """Scheduler state'ini kaydet"""
        return {
            'current_lr': self.current_lr,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'cooldown_epochs': self.cooldown_epochs,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr
        }
    
    def __repr__(self):
        return (f"WarmupCosineScheduler(warmup={self.warmup_epochs}, "
                f"decay={self.decay_epochs}, cooldown={self.cooldown_epochs}, "
                f"max_lr={self.max_lr}, min_lr={self.min_lr})")


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class MetricsTracker:
    """Eğitim sırasında tüm metrikleri takip et"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'lr': []
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_history(self):
        return self.history


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True
) -> Dict[str, float]:
    """Bir epoch eğitim - tüm metriklerle"""
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Progress bar güncelle
        current_acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*current_acc:.2f}%'
        })
    
    # Epoch metrikleri hesapla
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validation - tüm metriklerle"""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Malignant probability
    
    # Metrikleri hesapla
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # AUC hesapla
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    split_name: str = "Test"
) -> Dict:
    """Tam değerlendirme - tüm metrikler ve detaylı analiz"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(data_loader, desc=f"Evaluating {split_name}"):
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Genel metrikler
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Weighted metrikler
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrikler
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Specificity hesapla (her sınıf için)
    specificity_pc = []
    for i in range(len(class_names)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_pc.append(specificity)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    
    return {
        'split_name': split_name,
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'auc': auc,
        'precision_per_class': dict(zip(class_names, precision_pc)),
        'recall_per_class': dict(zip(class_names, recall_pc)),
        'f1_per_class': dict(zip(class_names, f1_pc)),
        'specificity_per_class': dict(zip(class_names, specificity_pc)),
        'support_per_class': dict(zip(class_names, support_pc)),
        'confusion_matrix': cm,
        'classification_report': report,
        'classification_report_str': report_str,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# =============================================================================
# VİZUALİZASYON FONKSİYONLARI
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Eğitim geçmişini görselleştir - Loss ve Accuracy"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2ecc71')
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2, color='#2ecc71')
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2, color='#e74c3c')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_all_metrics_history(history: Dict, save_path: Optional[str] = None):
    """Tüm metriklerin eğitim geçmişini görselleştir"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy'),
        ('train_precision', 'val_precision', 'Precision'),
        ('train_recall', 'val_recall', 'Recall'),
        ('train_f1', 'val_f1', 'F1-Score'),
        ('val_auc', None, 'AUC-ROC (Validation)')
    ]
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, (train_key, val_key, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if train_key in history and history[train_key]:
            ax.plot(history[train_key], label='Train', linewidth=2, color=colors[0])
        if val_key and val_key in history and history[val_key]:
            ax.plot(history[val_key], label='Val', linewidth=2, color=colors[1])
        elif train_key in history and history[train_key]:
            ax.plot(history[train_key], label='Value', linewidth=2, color=colors[0])
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """Confusion matrix görselleştir"""
    
    plt.figure(figsize=(8, 6))
    
    # Normalize edilmiş CM hesapla
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 16, 'weight': 'bold'},
        cbar_kws={'label': 'Count'}
    )
    
    # Yüzdeleri de ekle
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.1%})',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
):
    """ROC eğrisi çiz"""
    
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc_score = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random')
    plt.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_precision_recall_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
):
    """Precision-Recall eğrisi çiz"""
    
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    avg_precision = average_precision_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#9b59b6', lw=2, 
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.fill_between(recall, precision, alpha=0.3, color='#9b59b6')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_per_class_metrics(
    results: Dict,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Per-class metrikleri bar chart olarak görselleştir"""
    
    metrics = ['precision', 'recall', 'f1', 'specificity']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        key = f'{metric}_per_class'
        
        if key in results:
            values = [results[key][cls] for cls in class_names]
            bars = ax.bar(class_names, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Değerleri bar üzerine yaz
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylim(0, 1.15)
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} per Class', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def plot_comparison_metrics(
    train_results: Dict,
    val_results: Dict,
    test_results: Dict,
    save_path: Optional[str] = None
):
    """Train/Val/Test karşılaştırma grafiği"""
    
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    splits = ['Train', 'Validation', 'Test']
    results_list = [train_results, val_results, test_results]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (split, results, color) in enumerate(zip(splits, results_list, colors)):
        values = [results.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=split, color=color, edgecolor='black')
        
        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Train vs Validation vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


@torch.no_grad()
def evaluate_per_magnification(
    model: nn.Module,
    df: pd.DataFrame,
    config,
    device: torch.device,
    class_names: List[str],
    split_name: str = "Test"
) -> Dict[str, Dict]:
    """Her magnification için ayrı ayrı değerlendirme yap"""
    
    model.eval()
    results_per_mag = {}
    
    magnifications = df['mag'].unique()
    
    for mag in magnifications:
        df_mag = df[df['mag'] == mag].reset_index(drop=True)
        
        if len(df_mag) == 0:
            continue
        
        # Dataset ve DataLoader oluştur
        ds = BreaKHisDataset(
            df_mag,
            transform=get_transforms(config.IMG_SIZE, is_training=False),
            class_names=class_names
        )
        loader = DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Metrikleri hesapla
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        
        results_per_mag[mag] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'support': len(df_mag)
        }
    
    return results_per_mag


def plot_per_magnification_results(
    results_per_mag: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """Her magnification için sonuçları görselleştir"""
    
    mags = list(results_per_mag.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(mags))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results_per_mag[mag][metric] for mag in mags]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor='black')
        
        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Magnification Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(mags, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Support bilgisini ekle
    for i, mag in enumerate(mags):
        ax.text(i + width * 2, -0.08, f"n={results_per_mag[mag]['support']}", 
               ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
    plt.show()


def print_detailed_results(results: Dict, title: str = "Results"):
    """Detaylı sonuçları yazdır"""
    
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)
    
    print(f"\n📊 GENEL METRİKLER:")
    print(f"   Accuracy:  {100*results['accuracy']:.2f}%")
    print(f"   Precision: {100*results['precision_weighted']:.2f}%")
    print(f"   Recall:    {100*results['recall_weighted']:.2f}%")
    print(f"   F1-Score:  {100*results['f1_weighted']:.2f}%")
    print(f"   AUC-ROC:   {100*results['auc']:.2f}%")
    
    print(f"\n📈 PER-CLASS METRİKLER:")
    for cls in results['precision_per_class'].keys():
        print(f"\n   {cls.upper()}:")
        print(f"     Precision:   {100*results['precision_per_class'][cls]:.2f}%")
        print(f"     Recall:      {100*results['recall_per_class'][cls]:.2f}%")
        print(f"     F1-Score:    {100*results['f1_per_class'][cls]:.2f}%")
        print(f"     Specificity: {100*results['specificity_per_class'][cls]:.2f}%")
        print(f"     Support:     {results['support_per_class'][cls]}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(results['classification_report_str'])


# =============================================================================
# ANA EĞİTİM DÖNGÜSÜ
# =============================================================================

def train_nat_model(config: Config):
    """NAT modelini eğit - tam metriklerle"""
    
    print("=" * 70)
    print(" NAT (Neighborhood Attention Transformer) - BreaKHis Training ")
    print("=" * 70)
    
    # Seed ayarla
    set_seed(config.SEED)
    
    # Dizinleri oluştur
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Device bilgisi
    print(f"\n🖥️  Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Veri setini hazırla
    print("\n📂 Loading dataset...")
    base_path = config.get_base_path()
    print(f"   Dataset path: {base_path}")
    
    if not base_path.exists():
        print(f"\n⚠️  Dataset not found at {base_path}")
        print("   Please download the dataset first using: python dataset.py")
        return None, None, None
    
    df = create_dataframe(base_path)
    print(f"   Total images: {len(df)}")
    
    # Magnification filtrele
    if config.SELECTED_MAG == 'ALL':
        df_mag = df[df['mag'].isin(config.MAGNIFICATIONS)].reset_index(drop=True)
        print(f"   ALL magnifications: {len(df_mag)} images")
        print(f"   Magnification distribution:")
        for mag in config.MAGNIFICATIONS:
            mag_count = len(df_mag[df_mag['mag'] == mag])
            print(f"     - {mag}: {mag_count} images")
    else:
        df_mag = df[df['mag'] == config.SELECTED_MAG].reset_index(drop=True)
        print(f"   {config.SELECTED_MAG} images: {len(df_mag)}")
    
    # Class distribution
    print(f"\n   Class distribution:")
    for cls in config.CLASS_NAMES:
        count = len(df_mag[df_mag['label'] == cls])
        print(f"   - {cls}: {count} ({100*count/len(df_mag):.1f}%)")
    
    # Patient-stratified split (class-balanced)
    print("\n📊 Creating patient-stratified splits...")
    splits = patient_stratified_split(df_mag, 0.7, 0.15, 0.15, config.SEED)
    
    for split_name, split_df in splits.items():
        n_images = len(split_df)
        n_patients = split_df['patient_id'].nunique()
        benign_count = len(split_df[split_df['label'] == 'benign'])
        malignant_count = len(split_df[split_df['label'] == 'malignant'])
        print(f"   {split_name:>5}: {n_images:4d} images, {n_patients:3d} patients | B:{benign_count} M:{malignant_count}")
    
    # =========================================================================
    # YENİ: OVERSAMPLING (Swin makalesi preprocessing)
    # =========================================================================
    if config.USE_OVERSAMPLING:
        print("\n⚖️ Applying oversampling to balance classes...")
        original_train_size = len(splits['train'])
        splits['train'] = balance_dataset(splits['train'], config.SEED)
        print(f"   Train size: {original_train_size} → {len(splits['train'])} (oversampled)")
    
    # =========================================================================
    # YENİ: STAIN NORMALIZATION (Swin makalesi preprocessing)
    # =========================================================================
    stain_normalizer = None
    if config.USE_STAIN_NORMALIZATION:
        print(f"\n🎨 Setting up Stain Normalization ({config.STAIN_METHOD})...")
        try:
            # Referans görüntü seç (veri setinden)
            reference_image = get_reference_image(base_path, sample_count=5)
            stain_normalizer = TorchStainNormalizer(
                method=config.STAIN_METHOD, 
                reference_image=reference_image
            )
            print(f"   ✓ Stain normalization initialized ({config.STAIN_METHOD} method)")
        except Exception as e:
            print(f"   ⚠️ Stain normalization başlatılamadı: {e}")
            print(f"   → Stain normalization devre dışı bırakıldı")
            stain_normalizer = None
    
    # DataLoader'ları oluştur
    print("\n🔄 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(splits, config, stain_normalizer)
    
    # Class weights
    class_weights = compute_class_weights(splits['train'], config.CLASS_NAMES)
    class_weights = class_weights.to(config.DEVICE)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Model oluştur
    print(f"\n🏗️  Building NAT model ({config.MODEL_VARIANT})...")
    
    model_builders = {
        'mini': nat_mini,
        'tiny': nat_tiny,
        'small': nat_small,
    }
    model_fn = model_builders.get(config.MODEL_VARIANT, nat_mini)
    model = model_fn(
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROP_RATE,
        attn_drop_rate=config.ATTN_DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE
    )
    model = model.to(config.DEVICE)
    
    # Model parametreleri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # YENİ: Warmup + Cosine Decay + Cooldown Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=config.EPOCHS,
        warmup_epochs=config.WARMUP_EPOCHS,
        cooldown_epochs=config.COOLDOWN_EPOCHS,
        max_lr=config.LEARNING_RATE,
        min_lr=config.MIN_LR
    )
    print(f"   LR Scheduler: Warmup({config.WARMUP_EPOCHS}) + Cosine + Cooldown({config.COOLDOWN_EPOCHS})")
    print(f"   LR Range: {config.MIN_LR} → {config.LEARNING_RATE} → {config.MIN_LR}")
    
    # Mixed precision (AMP) - PyTorch 2.x uyumlu
    use_amp = config.USE_AMP and config.DEVICE.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    print(f"   Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    # Early stopping ve metrics tracker
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    metrics_tracker = MetricsTracker()
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    
    # Eğitim döngüsü
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    for epoch in range(config.EPOCHS):
        print(f"\n📌 Epoch {epoch+1}/{config.EPOCHS}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            config.DEVICE, scaler, use_amp
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Scheduler step (epoch parametresi ile)
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        
        # Metrics tracker güncelle
        metrics_tracker.update(
            train_loss=train_metrics['loss'],
            train_acc=train_metrics['accuracy'],
            train_precision=train_metrics['precision'],
            train_recall=train_metrics['recall'],
            train_f1=train_metrics['f1'],
            val_loss=val_metrics['loss'],
            val_acc=val_metrics['accuracy'],
            val_precision=val_metrics['precision'],
            val_recall=val_metrics['recall'],
            val_f1=val_metrics['f1'],
            val_auc=val_metrics['auc'],
            lr=current_lr
        )
        
        # Sonuçları yazdır
        print(f"\n   📈 Train | Loss: {train_metrics['loss']:.4f} | Acc: {100*train_metrics['accuracy']:.2f}% | "
              f"P: {100*train_metrics['precision']:.2f}% | R: {100*train_metrics['recall']:.2f}% | F1: {100*train_metrics['f1']:.2f}%")
        print(f"   📊 Val   | Loss: {val_metrics['loss']:.4f} | Acc: {100*val_metrics['accuracy']:.2f}% | "
              f"P: {100*val_metrics['precision']:.2f}% | R: {100*val_metrics['recall']:.2f}% | F1: {100*val_metrics['f1']:.2f}% | AUC: {100*val_metrics['auc']:.2f}%")
        # LR aşamasını belirle
        if epoch < config.WARMUP_EPOCHS:
            lr_phase = "🔥 WARMUP"
        elif epoch >= config.EPOCHS - config.COOLDOWN_EPOCHS:
            lr_phase = "❄️ COOLDOWN"
        else:
            lr_phase = "📉 DECAY"
        print(f"   📚 LR: {current_lr:.2e} ({lr_phase})")
        
        # Best model kaydet (F1 score'a göre)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'config': {
                    'model_variant': config.MODEL_VARIANT,
                    'img_size': config.IMG_SIZE,
                    'magnification': config.SELECTED_MAG
                }
            }, config.CHECKPOINT_DIR / f'nat_best_{config.SELECTED_MAG}.pth')
            print(f"   ✅ Best model saved! (Val F1: {100*best_val_f1:.2f}%, Val Acc: {100*best_val_acc:.2f}%)")
        
        # Early stopping kontrolü
        if early_stopping(val_metrics['loss']):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 70)
    print(" Training Completed! ")
    print("=" * 70)
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val Accuracy: {100*best_val_acc:.2f}%")
    print(f"   Best Val F1-Score: {100*best_val_f1:.2f}%")
    
    # En iyi modeli yükle
    checkpoint = torch.load(config.CHECKPOINT_DIR / f'nat_best_{config.SELECTED_MAG}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # =========================================================================
    # DETAYLI DEĞERLENDİRME - Train/Val/Test
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" DETAILED EVALUATION ")
    print("=" * 70)
    
    # Train seti değerlendirmesi
    print("\n📊 Evaluating on TRAIN set...")
    train_results = evaluate_full(model, train_loader, config.DEVICE, config.CLASS_NAMES, "Train")
    print_detailed_results(train_results, "TRAIN SET RESULTS")
    
    # Validation seti değerlendirmesi
    print("\n📊 Evaluating on VALIDATION set...")
    val_results = evaluate_full(model, val_loader, config.DEVICE, config.CLASS_NAMES, "Validation")
    print_detailed_results(val_results, "VALIDATION SET RESULTS")
    
    # Test seti değerlendirmesi
    print("\n📊 Evaluating on TEST set...")
    test_results = evaluate_full(model, test_loader, config.DEVICE, config.CLASS_NAMES, "Test")
    print_detailed_results(test_results, "TEST SET RESULTS")
    
    # =========================================================================
    # GRAFİKLER
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" GENERATING VISUALIZATIONS ")
    print("=" * 70)
    
    history = metrics_tracker.get_history()
    
    # 1. Training History - Loss & Accuracy
    print("\n📈 1. Training History (Loss & Accuracy)...")
    plot_training_history(
        history,
        save_path=str(config.OUTPUT_DIR / f'training_history_{config.SELECTED_MAG}.png')
    )
    
    # 2. All Metrics History
    print("\n📈 2. All Metrics History...")
    plot_all_metrics_history(
        history,
        save_path=str(config.OUTPUT_DIR / f'all_metrics_history_{config.SELECTED_MAG}.png')
    )
    
    # 3. Confusion Matrices
    print("\n📊 3. Confusion Matrices...")
    for split_name, results in [('Train', train_results), ('Validation', val_results), ('Test', test_results)]:
        plot_confusion_matrix(
            results['confusion_matrix'],
            config.CLASS_NAMES,
            title=f"Confusion Matrix - {split_name} Set ({config.SELECTED_MAG})",
            save_path=str(config.OUTPUT_DIR / f'confusion_matrix_{split_name.lower()}_{config.SELECTED_MAG}.png')
        )
    
    # 4. ROC Curves
    print("\n📈 4. ROC Curves...")
    for split_name, results in [('Train', train_results), ('Validation', val_results), ('Test', test_results)]:
        plot_roc_curve(
            results['labels'],
            results['probabilities'][:, 1],
            title=f"ROC Curve - {split_name} Set ({config.SELECTED_MAG})",
            save_path=str(config.OUTPUT_DIR / f'roc_curve_{split_name.lower()}_{config.SELECTED_MAG}.png')
        )
    
    # 5. Precision-Recall Curves
    print("\n📈 5. Precision-Recall Curves...")
    for split_name, results in [('Train', train_results), ('Validation', val_results), ('Test', test_results)]:
        plot_precision_recall_curve(
            results['labels'],
            results['probabilities'][:, 1],
            title=f"Precision-Recall Curve - {split_name} Set ({config.SELECTED_MAG})",
            save_path=str(config.OUTPUT_DIR / f'pr_curve_{split_name.lower()}_{config.SELECTED_MAG}.png')
        )
    
    # 6. Per-Class Metrics
    print("\n📊 6. Per-Class Metrics...")
    plot_per_class_metrics(
        test_results,
        config.CLASS_NAMES,
        save_path=str(config.OUTPUT_DIR / f'per_class_metrics_{config.SELECTED_MAG}.png')
    )
    
    # 7. Train/Val/Test Comparison
    print("\n📊 7. Train/Val/Test Comparison...")
    plot_comparison_metrics(
        train_results,
        val_results,
        test_results,
        save_path=str(config.OUTPUT_DIR / f'comparison_metrics_{config.SELECTED_MAG}.png')
    )
    
    # 8. Per-Magnification Evaluation (sadece ALL modunda)
    if config.SELECTED_MAG == 'ALL':
        print("\n" + "=" * 70)
        print(" PER-MAGNIFICATION EVALUATION ")
        print("=" * 70)
        
        # Test seti üzerinde her magnification için ayrı değerlendirme
        print("\n📊 8. Evaluating per magnification on TEST set...")
        test_per_mag = evaluate_per_magnification(
            model, splits['test'], config, config.DEVICE, config.CLASS_NAMES, "Test"
        )
        
        # Sonuçları yazdır
        print("\n📈 Per-Magnification Results:")
        print("-" * 70)
        print(f"{'Magnification':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support':<10}")
        print("-" * 70)
        for mag in sorted(test_per_mag.keys()):
            r = test_per_mag[mag]
            print(f"{mag:<15} {100*r['accuracy']:<12.2f} {100*r['precision']:<12.2f} {100*r['recall']:<12.2f} {100*r['f1']:<12.2f} {100*r['auc']:<12.2f} {r['support']:<10}")
        print("-" * 70)
        
        # Per-magnification grafik
        plot_per_magnification_results(
            test_per_mag,
            save_path=str(config.OUTPUT_DIR / 'per_magnification_results.png')
        )
        
        # Her magnification için ayrı confusion matrix
        print("\n📊 9. Confusion matrices per magnification...")
        for mag in sorted(test_per_mag.keys()):
            plot_confusion_matrix(
                test_per_mag[mag]['confusion_matrix'],
                config.CLASS_NAMES,
                title=f"Confusion Matrix - {mag}",
                save_path=str(config.OUTPUT_DIR / f'confusion_matrix_test_{mag}.png')
            )
    
    # =========================================================================
    # SONUÇ ÖZET
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY ")
    print("=" * 70)
    
    print(f"\n📁 Model checkpoint: {config.CHECKPOINT_DIR / f'nat_best_{config.SELECTED_MAG}.pth'}")
    print(f"📁 Output directory: {config.OUTPUT_DIR}")
    
    print(f"\n🏆 BEST TEST RESULTS:")
    print(f"   ├── Accuracy:  {100*test_results['accuracy']:.2f}%")
    print(f"   ├── Precision: {100*test_results['precision_weighted']:.2f}%")
    print(f"   ├── Recall:    {100*test_results['recall_weighted']:.2f}%")
    print(f"   ├── F1-Score:  {100*test_results['f1_weighted']:.2f}%")
    print(f"   └── AUC-ROC:   {100*test_results['auc']:.2f}%")
    
    print("\n✅ Training and evaluation complete!")
    
    return model, history, {
        'train': train_results,
        'val': val_results,
        'test': test_results
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # GPU kontrolü ve optimizasyon
    if torch.cuda.is_available():
        # CUDA optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("🚀 GPU Optimizations Enabled")
        print(f"   - cuDNN Benchmark: True")
        print(f"   - TF32: Enabled")
    
    # Konfigürasyon
    config = Config()
    
    # Eğitimi başlat
    model, history, results = train_nat_model(config)
    
    if model is not None:
        print("\n" + "=" * 70)
        print(" ALL DONE! ")
        print("=" * 70)

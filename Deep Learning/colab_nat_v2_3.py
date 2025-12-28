"""
NAT BreaKHis V2 - Google Colab Script (NATTEN Optimized)
========================================================

Colab'da Kullanım:
1. Önce NATTEN kur: !pip install natten -f https://shi-labs.com/natten/wheels/cu121/torch2.1/index.html
2. Runtime > Change runtime type > GPU (A100)
3. Bu kodu çalıştır
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import random
import re
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Multiprocessing hatalarını kapat
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)

# NATTEN Import ve API Test
USE_NATTEN = False
NATTEN_NA2D = None
NATTEN_API_STYLE = None  # 'new' veya 'old'

try:
    from natten import NeighborhoodAttention2D as _NATTEN_NA2D
    import inspect
    
    # NATTEN API'sini test et
    sig = inspect.signature(_NATTEN_NA2D.__init__)
    params = list(sig.parameters.keys())
    print(f"🔍 NATTEN API parametreleri: {params}")
    
    # Test ile doğru API'yi bul
    try:
        # Yeni API testi (0.21.x) - 'attn_drop' yerine farklı isim olabilir
        test_layer = _NATTEN_NA2D(64, num_heads=2, kernel_size=7)
        del test_layer
        NATTEN_NA2D = _NATTEN_NA2D
        USE_NATTEN = True
        NATTEN_API_STYLE = 'new'
        print("✅ NATTEN yüklendi - GPU hızlandırması aktif! (Yeni API)")
    except Exception as e:
        print(f"⚠️ NATTEN API test hatası: {e}")
        USE_NATTEN = False
        
except ImportError:
    print("⚠️ NATTEN bulunamadı! Yavaş implementasyon kullanılacak.")
    print("   Kurulum: !pip install natten")

# =============================================================================
# DATASET DOWNLOAD
# =============================================================================

print("📥 Downloading BreaKHis dataset...")
import kagglehub
dataset_path = kagglehub.dataset_download("ambarish/breakhis")
print(f"✅ Dataset path: {dataset_path}")

# =============================================================================
# CONFIG
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")

class Config:
    BASE_PATH = Path(dataset_path) / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
    SEED = 42
    IMG_SIZE = 224
    
    # NATTEN yoksa fallback çok bellek kullanır, batch küçültüldü
    BATCH_SIZE = 256                  # Artırıldı (NATTEN ile mümkün)
    NUM_WORKERS = 8                   # Azaltıldı (multiprocessing hata önleme)
    EPOCHS = 50                       # Artırıldı (daha uzun eğitim)
    LEARNING_RATE = 2e-4              # Artırıldı
    WARMUP_EPOCHS = 10                # Artırıldı
    COOLDOWN_EPOCHS = 5               # Artırıldı
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01               # Azaltıldı
    
    NUM_CLASSES = 2
    CLASS_NAMES = ['benign', 'malignant']
    DROP_RATE = 0.3                   # Artırıldı
    ATTN_DROP_RATE = 0.1
    DROP_PATH_RATE = 0.1              # Azaltıldı
    
    # === İYİLEŞTİRMELER ===
    USE_STAIN_NORMALIZATION = False   # Çok yavaş, kapalı bırak
    USE_OVERSAMPLING = False
    USE_UNDERSAMPLING = True          # Yeni: Undersampling
    # criterion burada tanımlanmıyor - class weights hesaplandıktan sonra tanımlanacak (satır 499)

    PATIENCE = 15                     # Artırıldı
    USE_AMP = True
    DEVICE = DEVICE
    
    # Yeni iyileştirmeler
    LABEL_SMOOTHING = 0.1              # Artırıldı
    USE_TTA = True                    # Test-Time Augmentation
    USE_MIXUP = True                  # Mixup augmentation
    USE_CUTMIX = True                 # Yeni: CutMix
    USE_MIXUP_OR_CUTMIX = True        # Yeni: Her batch'te birini seç
    MIXUP_ALPHA = 0.2                 # Mixup alpha (0.2-0.4 önerilen)
    CUTMIX_ALPHA = 1.0                # Yeni: CutMix alpha

config = Config()
print(f"📊 Config: Batch={config.BATCH_SIZE}, Epochs={config.EPOCHS}, LR={config.LEARNING_RATE}")

# =============================================================================
# STAIN NORMALIZATION (Kapalı ama kod duruyor)
# =============================================================================

class StainNormalizer:
    def __init__(self):
        self.stain_matrix_ref = None
        self.max_conc_ref = None
    
    def fit(self, ref_img):
        if ref_img.max() <= 1.0:
            ref_img = (ref_img * 255).astype(np.uint8)
        self.stain_matrix_ref, self.max_conc_ref = self._get_params(ref_img)
    
    def transform(self, img):
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return self._normalize(img)
    
    def _rgb_to_od(self, img):
        return -np.log((img.astype(np.float32) + 1) / 255.0)
    
    def _od_to_rgb(self, od):
        return np.clip(255 * np.exp(-od), 0, 255).astype(np.uint8)
    
    def _get_params(self, img):
        od = self._rgb_to_od(img).reshape(-1, 3)
        mask = np.all(od > 0.15, axis=1)
        od_f = od[mask]
        if len(od_f) < 100:
            return np.array([[0.644, 0.717, 0.267], [0.093, 0.954, 0.283]]), np.array([1.0, 1.0])
        try:
            _, _, V = np.linalg.svd(od_f, full_matrices=False)
            V = V[:2, :]
            phi = np.arctan2(V[1, :], V[0, :])
            v1 = np.array([np.cos(np.percentile(phi, 1)), np.sin(np.percentile(phi, 1))])
            v2 = np.array([np.cos(np.percentile(phi, 99)), np.sin(np.percentile(phi, 99))])
            sm = np.array([V.T @ v1, V.T @ v2])
            sm = sm / np.linalg.norm(sm, axis=1, keepdims=True)
            conc = np.linalg.lstsq(sm.T, od_f.T, rcond=None)[0]
            return sm, np.percentile(conc, 99, axis=1)
        except:
            return np.array([[0.644, 0.717, 0.267], [0.093, 0.954, 0.283]]), np.array([1.0, 1.0])
    
    def _normalize(self, img):
        sm_src, mc_src = self._get_params(img)
        od = self._rgb_to_od(img)
        h, w, _ = od.shape
        od = od.reshape(-1, 3)
        try:
            conc = np.linalg.lstsq(sm_src.T, od.T, rcond=None)[0]
            conc = conc / (mc_src.reshape(-1, 1) + 1e-6) * self.max_conc_ref.reshape(-1, 1)
            od_norm = (self.stain_matrix_ref.T @ conc).T.reshape(h, w, 3)
            return self._od_to_rgb(od_norm)
        except:
            return img


class TorchStainNorm:
    def __init__(self, ref_img=None):
        self.norm = StainNormalizer()
        self.fitted = False
        if ref_img is not None:
            self.fit(ref_img)
    
    def fit(self, ref_img):
        if isinstance(ref_img, (str, Path)):
            ref_img = np.array(Image.open(ref_img).convert('RGB'))
        elif isinstance(ref_img, Image.Image):
            ref_img = np.array(ref_img.convert('RGB'))
        self.norm.fit(ref_img)
        self.fitted = True
    
    def __call__(self, img):
        if not self.fitted:
            return img
        img_np = np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img
        return Image.fromarray(self.norm.transform(img_np))

# =============================================================================
# NAT MODEL - NATTEN OPTIMIZED
# =============================================================================

class NeighborhoodAttention2D(nn.Module):
    """NATTEN varsa GPU-optimized, yoksa fallback kullanır"""
    def __init__(self, dim, num_heads, kernel_size=7, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.use_natten = USE_NATTEN
        
        if self.use_natten and NATTEN_NA2D is not None:
            # NATTEN 0.21.x - sadece temel parametreler kullan
            # (dropout parametreleri sonra elle eklenecek)
            self.attn = NATTEN_NA2D(
                dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
            )
            # Dropout'ları ayrı ekle
            self.attn_drop_layer = nn.Dropout(attn_drop)
            self.proj_drop_layer = nn.Dropout(proj_drop)
        else:
            self.use_natten = False
            self._init_fallback(dim, num_heads, kernel_size, attn_drop, proj_drop)
    
    def _init_fallback(self, dim, num_heads, kernel_size, attn_drop, proj_drop):
        """Fallback implementasyonu için gerekli katmanları oluştur"""
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.padding = kernel_size // 2
        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size * kernel_size))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        self.attn_drop_layer = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        if self.use_natten:
            # NATTEN + manuel dropout
            out = self.attn(x)
            return self.proj_drop_layer(out)
        else:
            B, H, W, C = x.shape
            qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 5, 1, 2)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            k = F.pad(k, (self.padding,) * 4).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
            v = F.pad(v, (self.padding,) * 4).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
            k = k.reshape(B, self.num_heads, self.head_dim, H, W, -1).permute(0, 1, 3, 4, 2, 5)
            v = v.reshape(B, self.num_heads, self.head_dim, H, W, -1).permute(0, 1, 3, 4, 2, 5)
            q = q.permute(0, 1, 3, 4, 2).unsqueeze(-2)
            attn = F.softmax(q @ k + self.rpb.view(1, self.num_heads, 1, 1, 1, -1), dim=-1)
            out = (self.attn_drop_layer(attn) @ v.transpose(-2, -1)).squeeze(-2).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
            return self.proj_drop(self.proj(out))

class Mlp(nn.Module):
    def __init__(self, dim, hidden=None, drop=0.0):
        super().__init__()
        hidden = hidden or dim * 4
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.act, self.drop = nn.GELU(), nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        kp = 1 - self.p
        mask = kp + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype)
        return x.div(kp) * mask.floor_()

class NATBlock(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = NeighborhoodAttention2D(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class PatchEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 4, 4)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))

class PatchMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.red = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    def forward(self, x):
        B, H, W, C = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))

class NATStage(nn.Module):
    def __init__(self, dim, depth, heads, drop=0.0, attn_drop=0.0, drop_path=None, downsample=True):
        super().__init__()
        dp = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList([NATBlock(dim, heads, drop, attn_drop, dp[i]) for i in range(depth)])
        self.down = PatchMerge(dim) if downsample else None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.down(x) if self.down else x

class NAT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64, depths=[3,4,6,5], heads=[2,4,8,16], drop=0.0, attn_drop=0.0, drop_path=0.2):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList([
            NATStage(embed_dim * 2**i, depths[i], heads[i], drop, attn_drop, 
                     dpr[sum(depths[:i]):sum(depths[:i+1])], i < len(depths)-1)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim * 2**(len(depths)-1))
        self.head = nn.Linear(embed_dim * 2**(len(depths)-1), num_classes)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.patch_embed(x)
        for s in self.stages: x = s(x)
        x = self.norm(x).permute(0, 3, 1, 2).flatten(2).mean(-1)
        return self.head(x)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def parse_path(p):
    fname = os.path.basename(p)
    parts = fname.split('_')
    label = 'benign' if parts[1].upper().startswith('B') else 'malignant'
    mag = Path(p).parents[0].name.upper()
    try: patient = parts[2].rsplit('-', 2)[0]
    except: patient = fname
    return label, mag, patient

def create_df(base):
    paths = sorted([str(p) for p in Path(base).rglob('*.png')])
    return pd.DataFrame([{'filepath': p, 'label': parse_path(p)[0], 'mag': parse_path(p)[1], 'patient_id': parse_path(p)[2]} for p in paths])

def split_data(df, seed=42):
    train, val, test = [], [], []
    for lbl in df['label'].unique():
        df_l = df[df['label'] == lbl]
        pts = df_l['patient_id'].unique().tolist()
        random.Random(seed).shuffle(pts)
        n = len(pts)
        n_tr, n_val = int(0.7 * n), int(0.15 * n)
        train.append(df_l[df_l['patient_id'].isin(set(pts[:n_tr]))])
        val.append(df_l[df_l['patient_id'].isin(set(pts[n_tr:n_tr+n_val]))])
        test.append(df_l[df_l['patient_id'].isin(set(pts[n_tr+n_val:]))])
    return {'train': pd.concat(train).reset_index(drop=True), 'val': pd.concat(val).reset_index(drop=True), 'test': pd.concat(test).reset_index(drop=True)}

def oversample(df, seed=42):
    counts = df['label'].value_counts()
    mx = counts.max()
    dfs = []
    for lbl in df['label'].unique():
        dl = df[df['label'] == lbl]
        if len(dl) < mx:
            dl = pd.concat([dl, dl.sample(n=mx-len(dl), replace=True, random_state=seed)])
            print(f"   ⚖️ {lbl}: {counts[lbl]} → {len(dl)}")
        dfs.append(dl)
    return pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

def balance_sample(df, seed=42):
    """Her sınıftan eşit sayıda örnek al (undersampling)"""
    counts = df['label'].value_counts()
    min_count = counts.min()
    dfs = []
    for lbl in df['label'].unique():
        dl = df[df['label'] == lbl]
        if len(dl) > min_count:
            dl = dl.sample(n=min_count, random_state=seed)
            print(f"   ⚖️ {lbl}: {counts[lbl]} → {len(dl)} (undersampled)")
        else:
            print(f"   ⚖️ {lbl}: {counts[lbl]} (kept)")
        dfs.append(dl)
    return pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

# Prepare data
df = create_df(config.BASE_PATH)
print(f"📊 Total: {len(df)}, Benign: {len(df[df['label']=='benign'])}, Malignant: {len(df[df['label']=='malignant'])}")

splits = split_data(df)
if config.USE_UNDERSAMPLING:
    print("⚖️ Balanced Sampling (Undersampling)...")
    splits['train'] = balance_sample(splits['train'])
elif config.USE_OVERSAMPLING:
    print("⚖️ Oversampling...")
    splits['train'] = oversample(splits['train'])

# Stain normalization
stain_norm = None
if config.USE_STAIN_NORMALIZATION:
    print("🎨 Setting up stain normalization...")
    ref = list(config.BASE_PATH.rglob('*.png'))[0]
    stain_norm = TorchStainNorm(ref)
    print("   ✅ Done!")

# =============================================================================
# TRANSFORMS & DATASET
# =============================================================================

class RandomSharp:
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.0, 2.5))
        return img

# RandAugment desteği
try:
    from timm.data.auto_augment import rand_augment_transform
    USE_RANDAUGMENT = True
except:
    USE_RANDAUGMENT = False
    print("⚠️ timm bulunamadı, RandAugment kullanılamayacak")

def get_tfm(train=True, sn=None):
    tfms = [sn] if sn else []
    if train:
        # Standart augmentation (RandAugment her örnek için ayrı uygulanacak şekilde Dataset içinde)
        tfms += [T.Resize((272, 272)), T.RandomCrop(224), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                 T.RandomRotation(30), RandomSharp(), T.ColorJitter(0.3, 0.3, 0.2, 0.1),
                 T.GaussianBlur(3), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 T.RandomErasing(p=0.2)]
    else:
        tfms += [T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(tfms)

class DS(Dataset):
    def __init__(self, df, tfm, use_randaugment=False):
        self.df, self.tfm = df.reset_index(drop=True), tfm
        self.c2i = {'benign': 0, 'malignant': 1}
        self.use_randaugment = use_randaugment and USE_RANDAUGMENT
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r['filepath']).convert('RGB')
        
        # %50 ihtimalle RandAugment kullan
        if self.use_randaugment and random.random() < 0.5:
            try:
                # RandAugment transform
                img = T.Resize((272, 272))(img)
                img = T.RandomCrop(224)(img)
                img = rand_augment_transform('rand-m9-mstd0.5-inc1', {})(img)
                img = T.ToTensor()(img)
                img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                return img, self.c2i[r['label']]
            except:
                pass
        
        return self.tfm(img), self.c2i[r['label']]

# NUM_WORKERS azaltıldı (multiprocessing hata önleme)
# persistent_workers kaldırıldı (bazen takılma yapıyor)
train_dl = DataLoader(DS(splits['train'], get_tfm(True, stain_norm), use_randaugment=True), 
                     config.BATCH_SIZE, shuffle=True, 
                     num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
val_dl = DataLoader(DS(splits['val'], get_tfm(False, stain_norm)), config.BATCH_SIZE, 
                   num_workers=config.NUM_WORKERS, pin_memory=True)
test_dl = DataLoader(DS(splits['test'], get_tfm(False, stain_norm)), config.BATCH_SIZE, 
                    num_workers=config.NUM_WORKERS, pin_memory=True)

# Class weights
cc = splits['train']['label'].value_counts()
cw = torch.tensor([len(splits['train']) / (2 * cc.get(c, 1)) for c in ['benign', 'malignant']], dtype=torch.float32).to(DEVICE)
print(f"✅ DataLoaders ready! Train batches: {len(train_dl)}")

# =============================================================================
# SCHEDULER
# =============================================================================

class WarmupCosine:
    def __init__(self, opt, epochs, warmup=3, cooldown=2, max_lr=5e-4, min_lr=1e-6):
        self.opt, self.epochs, self.warmup, self.cooldown = opt, epochs, warmup, cooldown
        self.max_lr, self.min_lr = max_lr, min_lr
        self.decay = epochs - warmup - cooldown
        self._lr = [0.0]
    def step(self, e):
        if e < self.warmup:
            lr = self.max_lr * (e + 1) / self.warmup
        elif e >= self.epochs - self.cooldown:
            lr = self.min_lr
        else:
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * (e - self.warmup) / self.decay))
        self._lr = [lr]
        for pg in self.opt.param_groups: pg['lr'] = lr
    def get_last_lr(self): return self._lr

# =============================================================================
# TRAINING
# =============================================================================

model = NAT(num_classes=2, drop=config.DROP_RATE, attn_drop=config.ATTN_DROP_RATE, drop_path=config.DROP_PATH_RATE).to(DEVICE)
print(f"🏗️ NAT params: {sum(p.numel() for p in model.parameters()):,}")
print(f"🚀 NATTEN aktif: {USE_NATTEN}")

# =============================================================================
# PRETRAINED MODEL YÜKLEME (ImageNet)
# =============================================================================

USE_PRETRAINED = True  # Pretrained model kullan
PRETRAINED_URL = "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth"

if USE_PRETRAINED:
    print("📥 Pretrained model yükleniyor (ImageNet)...")
    try:
        import urllib.request
        import tempfile
        import os
        
        # Pretrained checkpoint'i indir
        checkpoint_path = tempfile.mktemp(suffix='.pth')
        urllib.request.urlretrieve(PRETRAINED_URL, checkpoint_path)
        
        # Checkpoint'i yükle
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Model state_dict'ini al (classification head hariç)
        pretrained_dict = checkpoint.get('model', checkpoint)
        
        # Kendi modelimizin state_dict'i
        model_dict = model.state_dict()
        
        # Pretrained ağırlıkları filtrele (classification head hariç)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'head' not in k and model_dict[k].shape == v.shape}
        
        # Eksik ağırlıkları göster
        missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
        if missing_keys:
            print(f"⚠️ Eksik ağırlıklar (yeni başlatılacak): {missing_keys}")
        
        # Pretrained ağırlıkları yükle
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Pretrained ağırlıklar yüklendi! ({len(pretrained_dict)}/{len(model_dict)} katman)")
        print("   ⚠️ Classification head sıfırdan eğitilecek (2 sınıf için)")
        
        # Geçici dosyayı sil
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"⚠️ Pretrained model yüklenemedi: {e}")
        print("   Model sıfırdan başlatılacak...")
        USE_PRETRAINED = False
else:
    print("🔄 Model sıfırdan başlatılıyor (pretrained yok)")

# Label smoothing eklendi
criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=config.LABEL_SMOOTHING)
print(f"🏷️ Label Smoothing: {config.LABEL_SMOOTHING}")
optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = WarmupCosine(optimizer, config.EPOCHS, config.WARMUP_EPOCHS, config.COOLDOWN_EPOCHS, config.LEARNING_RATE, config.MIN_LR)
scaler = GradScaler('cuda') if config.USE_AMP else None

# =============================================================================
# MIXUP & CUTMIX FUNCTIONS
# =============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup: iki görüntüyü ve etiketlerini karıştırır"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup için loss hesaplama"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    """CutMix için rastgele bbox hesaplama"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """CutMix: görüntülerin bir kısmını kesip yapıştır"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Lambda'yı gerçek kesilen alana göre ayarla
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix için loss hesaplama"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if config.USE_MIXUP_OR_CUTMIX:
    print(f"🔀 Mixup/CutMix aktif: Mixup alpha={config.MIXUP_ALPHA}, CutMix alpha={config.CUTMIX_ALPHA}")
elif config.USE_MIXUP:
    print(f"🔀 Mixup aktif: alpha={config.MIXUP_ALPHA}")
elif config.USE_CUTMIX:
    print(f"🔀 CutMix aktif: alpha={config.CUTMIX_ALPHA}")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [], 'lr': []}
best_f1, patience_cnt = 0.0, 0

print("\n🚀 Training...")
print("=" * 60)

for epoch in range(config.EPOCHS):
    phase = "🔥WARMUP" if epoch < config.WARMUP_EPOCHS else ("❄️COOL" if epoch >= config.EPOCHS - config.COOLDOWN_EPOCHS else "📉DECAY")
    
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"E{epoch+1}"):
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
        
        # Mixup veya CutMix uygula
        if config.USE_MIXUP_OR_CUTMIX:
            if random.random() < 0.5:  # %50 Mixup, %50 CutMix
                imgs, lbls_a, lbls_b, lam = mixup_data(imgs, lbls, config.MIXUP_ALPHA)
                use_mixup = True
            else:
                imgs, lbls_a, lbls_b, lam = cutmix_data(imgs, lbls, config.CUTMIX_ALPHA)
                use_mixup = False
        elif config.USE_MIXUP:
            imgs, lbls_a, lbls_b, lam = mixup_data(imgs, lbls, config.MIXUP_ALPHA)
            use_mixup = True
        elif config.USE_CUTMIX:
            imgs, lbls_a, lbls_b, lam = cutmix_data(imgs, lbls, config.CUTMIX_ALPHA)
            use_mixup = False
        else:
            use_mixup = False
            lbls_a, lbls_b, lam = lbls, None, 1.0
        
        optimizer.zero_grad(set_to_none=True)
        if config.USE_AMP:
            with autocast('cuda'):
                out = model(imgs)
                if config.USE_MIXUP_OR_CUTMIX or config.USE_MIXUP or config.USE_CUTMIX:
                    if use_mixup:
                        loss = mixup_criterion(criterion, out, lbls_a, lbls_b, lam)
                    else:
                        loss = cutmix_criterion(criterion, out, lbls_a, lbls_b, lam)
                else:
                    loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            if config.USE_MIXUP_OR_CUTMIX or config.USE_MIXUP or config.USE_CUTMIX:
                if use_mixup:
                    loss = mixup_criterion(criterion, out, lbls_a, lbls_b, lam)
                else:
                    loss = cutmix_criterion(criterion, out, lbls_a, lbls_b, lam)
            else:
                loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
        
        loss_sum += loss.item() * imgs.size(0)
        # Mixup/CutMix'ta accuracy hesaplama: orijinal etiketlerle karşılaştır
        if config.USE_MIXUP_OR_CUTMIX or config.USE_MIXUP or config.USE_CUTMIX:
            correct += (lam * (out.argmax(1) == lbls_a).float() + (1 - lam) * (out.argmax(1) == lbls_b).float()).sum().item()
        else:
            correct += (out.argmax(1) == lbls).sum().item()
        total += lbls.size(0)
    
    train_loss, train_acc = loss_sum / total, correct / total
    
    model.eval()
    val_preds, val_lbls, val_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            out = model(imgs.to(DEVICE, non_blocking=True))
            val_preds.extend(out.argmax(1).cpu().numpy())
            val_lbls.extend(lbls.numpy())
            val_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
    
    val_acc = accuracy_score(val_lbls, val_preds)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_lbls, val_preds, average='weighted', zero_division=0)
    try: val_auc = roc_auc_score(val_lbls, val_probs)
    except: val_auc = 0.0
    
    scheduler.step(epoch)
    lr = scheduler.get_last_lr()[0]
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_auc'].append(val_auc)
    history['lr'].append(lr)
    
    print(f"E{epoch+1}: Train={100*train_acc:.1f}% | Val={100*val_acc:.1f}%, Prec={100*val_prec:.1f}%, Rec={100*val_rec:.1f}%, F1={100*val_f1:.1f}%, AUC={100*val_auc:.1f}% | LR={lr:.2e} {phase}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_cnt = 0
        torch.save(model.state_dict(), 'nat_best.pth')
        
        # Magnification seviyelerine göre ayrı checkpoint
        val_df = splits['val'].reset_index(drop=True)
        val_preds_array = np.array(val_preds)
        val_lbls_array = np.array(val_lbls)
        
        for mag in ['40X', '100X', '200X', '400X']:
            mag_val_mask = val_df['mag'] == mag
            if mag_val_mask.sum() > 0:
                mag_val_indices = [i for i in range(len(val_lbls_array)) if mag_val_mask.iloc[i]]
                if len(mag_val_indices) > 0:
                    mag_val_lbls = val_lbls_array[mag_val_indices]
                    mag_val_preds = val_preds_array[mag_val_indices]
                    mag_val_acc = accuracy_score(mag_val_lbls, mag_val_preds)
                    
                    # Her magnification için en iyi modeli kaydet
                    mag_best_key = f'best_{mag}_acc'
                    if not hasattr(config, mag_best_key) or mag_val_acc > getattr(config, mag_best_key, 0.0):
                        setattr(config, mag_best_key, mag_val_acc)
                        torch.save(model.state_dict(), f'nat_best_{mag}.pth')
                        print(f"   ✅ Best {mag} saved! Acc={100*mag_val_acc:.2f}%")
        
        print(f"   ✅ Best saved! F1={100*best_f1:.2f}%")
    else:
        patience_cnt += 1
        if patience_cnt >= config.PATIENCE:
            print(f"⚠️ Early stop at E{epoch+1}")
            break

print(f"\n✅ Done! Best F1: {100*best_f1:.2f}%")

# =============================================================================
# EVALUATION WITH TTA (Test-Time Augmentation)
# =============================================================================

model.load_state_dict(torch.load('nat_best.pth', weights_only=True))
model.eval()

# TTA transforms

tta_transforms = [
    # 1️⃣ Orijinal
    T.Compose([T.Resize((224, 224)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    # 2️⃣ Horizontal Flip
    T.Compose([T.Resize((224, 224)),T.RandomHorizontalFlip(p=1.0),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    # 3️⃣ Vertical Flip
    T.Compose([T.Resize((224, 224)),T.RandomVerticalFlip(p=1.0),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    # 4️⃣ Rotate +90
    T.Compose([T.Resize((224, 224)),T.RandomRotation((90, 90)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    # 5️⃣ Rotate -90
    T.Compose([T.Resize((224, 224)),T.RandomRotation((-90, -90)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    # 6️⃣ Larger scale crop
    T.Compose([T.Resize((256, 256)),T.CenterCrop(224),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    ]



if config.USE_TTA:
    print("🔄 Test-Time Augmentation (TTA) aktif...")
    test_preds, test_lbls, test_probs = [], [], []
    
    # Her örnek için TTA
    test_df = splits['test'].reset_index(drop=True)
    for idx in tqdm(range(len(test_df)), desc="TTA Testing"):
        row = test_df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        
        # Stain normalization uygula
        if stain_norm is not None:
            img = stain_norm(img)
        
        # Tüm TTA transformlarını uygula ve ortala
        tta_outputs = []
        for tfm in tta_transforms:
            img_t = tfm(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_t)
                tta_outputs.append(torch.softmax(out, 1))
        
        # Ortalama al
        avg_prob = torch.stack(tta_outputs).mean(0).cpu().numpy()[0]
        test_probs.append(avg_prob)
        test_preds.append(avg_prob.argmax())
        test_lbls.append(0 if row['label'] == 'benign' else 1)
    
    test_preds = np.array(test_preds)
    test_lbls = np.array(test_lbls)
    test_probs = np.array(test_probs)
else:
    # Normal test (TTA yok)
    test_preds, test_lbls, test_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_dl, desc="Testing"):
            out = model(imgs.to(DEVICE, non_blocking=True))
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_lbls.extend(lbls.numpy())
            test_probs.extend(torch.softmax(out, 1).cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_lbls = np.array(test_lbls)
    test_probs = np.array(test_probs)

acc = accuracy_score(test_lbls, test_preds)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(test_lbls, test_preds, average='weighted', zero_division=0)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(test_lbls, test_preds, average='macro', zero_division=0)
prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(test_lbls, test_preds, average=None, zero_division=0)
auc = roc_auc_score(test_lbls, test_probs[:, 1])

print("\n" + "=" * 60)
print("🎯 TEST RESULTS" + (" (with TTA)" if config.USE_TTA else ""))
print("=" * 60)
print(f"\n📊 GENEL METRİKLER:")
print(f"  Accuracy:  {100*acc:.2f}%")
print(f"  Precision (Weighted): {100*prec_weighted:.2f}%")
print(f"  Recall (Weighted):    {100*rec_weighted:.2f}%")
print(f"  F1-Score (Weighted):   {100*f1_weighted:.2f}%")
print(f"  Precision (Macro):     {100*prec_macro:.2f}%")
print(f"  Recall (Macro):        {100*rec_macro:.2f}%")
print(f"  F1-Score (Macro):      {100*f1_macro:.2f}%")
print(f"  AUC-ROC:               {100*auc:.2f}%")

print(f"\n📋 SINIF BAZINDA METRİKLER:")
print(f"  {'Sınıf':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"  {'-'*50}")
for i, class_name in enumerate(['benign', 'malignant']):
    print(f"  {class_name:<12} {100*prec_per_class[i]:>10.2f}%  {100*rec_per_class[i]:>10.2f}%  {100*f1_per_class[i]:>10.2f}%")

print("=" * 60)

# Magnification seviyelerine göre ayrı değerlendirme
print("\n" + "=" * 60)
print("🔬 MAGNIFICATION LEVEL RESULTS")
print("=" * 60)

# Test setinden magnification bilgisini al
test_df = splits['test'].reset_index(drop=True)
magnification_results = {}

for mag in ['40X', '100X', '200X', '400X']:
    mag_mask = test_df['mag'] == mag
    if mag_mask.sum() > 0:
        mag_test_lbls = np.array([test_lbls[i] for i in range(len(test_lbls)) if mag_mask.iloc[i]])
        mag_test_preds = np.array([test_preds[i] for i in range(len(test_preds)) if mag_mask.iloc[i]])
        mag_test_probs = np.array([test_probs[i] for i in range(len(test_probs)) if mag_mask.iloc[i]])
        
        if len(mag_test_lbls) > 0:
            mag_acc = accuracy_score(mag_test_lbls, mag_test_preds)
            mag_prec, mag_rec, mag_f1, _ = precision_recall_fscore_support(mag_test_lbls, mag_test_preds, average='weighted', zero_division=0)
            try:
                mag_auc = roc_auc_score(mag_test_lbls, mag_test_probs[:, 1])
            except:
                mag_auc = 0.0
            
            magnification_results[mag] = {
                'acc': mag_acc,
                'prec': mag_prec,
                'rec': mag_rec,
                'f1': mag_f1,
                'auc': mag_auc,
                'count': len(mag_test_lbls)
            }
            
            print(f"\n{mag}:")
            print(f"  Accuracy:  {100*mag_acc:.2f}%")
            print(f"  Precision: {100*mag_prec:.2f}%")
            print(f"  Recall:    {100*mag_rec:.2f}%")
            print(f"  F1-Score:  {100*mag_f1:.2f}%")
            print(f"  AUC-ROC:   {100*mag_auc:.2f}%")
            print(f"  Samples:   {len(mag_test_lbls)}")

print("\n" + "=" * 60)

print("\n📋 Classification Report:")
print(classification_report(test_lbls, test_preds, target_names=['benign', 'malignant'], digits=4))

# =============================================================================
# PLOTS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(history['train_acc'], label='Train')
axes[0,0].plot(history['val_acc'], label='Val')
axes[0,0].set_title('Accuracy'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(history['val_f1'], color='green')
axes[0,1].set_title('Val F1-Score'); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(history['lr'], color='orange')
axes[1,0].set_title('Learning Rate'); axes[1,0].grid(True, alpha=0.3)

cm = confusion_matrix(test_lbls, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1], xticklabels=['benign','malignant'], yticklabels=['benign','malignant'])
axes[1,1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()

print("\n✅ Results saved to results.png")
"""
NAT (Neighborhood Attention Transformer) Implementation for BreaKHis Dataset
=============================================================================

Bu dosya, Neighborhood Attention Transformer (NAT) mimarisini içerir.
NAT, görüntü sınıflandırma için tasarlanmış etkili bir transformer mimarisidir.

Temel Kavramlar:
----------------
1. Neighborhood Attention (NA): Her token sadece yerel komşularıyla attention hesaplar
2. Hiyerarşik yapı: Farklı çözünürlüklerde özellik çıkarımı
3. Sliding window: Pencere kayması olmadan doğrudan komşuluk dikkati

Referans: "Neighborhood Attention Transformer" - Hassani et al., 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# NEIGHBORHOOD ATTENTION - TEMEL BİLEŞEN
# =============================================================================

class NeighborhoodAttention2D(nn.Module):
    """
    2D Neighborhood Attention Modülü
    
    Standart self-attention'dan farkı:
    - Her piksel sadece kernel_size x kernel_size komşuluğundaki piksellerle attention hesaplar
    - Bu, O(n²) yerine O(n * k²) karmaşıklık sağlar (k = kernel_size)
    
    Args:
        dim: Giriş kanalı sayısı
        num_heads: Attention head sayısı
        kernel_size: Komşuluk pencere boyutu (tek sayı olmalı)
        dilation: Dilasyon oranı (varsayılan 1)
        qkv_bias: Q, K, V projeksiyonlarında bias kullanılsın mı
        attn_drop: Attention dropout oranı
        proj_drop: Projeksiyon dropout oranı
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        assert kernel_size % 2 == 1, "Kernel size tek sayı olmalı"
        assert dim % num_heads == 0, "dim, num_heads'e tam bölünmeli"
        
        # Q, K, V projeksiyonları
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Relative positional bias (öğrenilebilir)
        # Her attention head için ayrı pozisyonel bias
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) formatında giriş tensörü
        Returns:
            (B, H, W, C) formatında çıkış tensörü
        """
        B, H, W, C = x.shape
        
        # Q, K, V hesapla
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, H, W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Neighborhood attention hesapla
        # Bu basitleştirilmiş bir implementasyon - gerçek NAT için natten kütüphanesi kullanılır
        attn_output = self._neighborhood_attention(q, k, v, H, W)
        
        # Projeksiyon
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        
        return x
    
    def _neighborhood_attention(self, q, k, v, H, W):
        """
        Basitleştirilmiş neighborhood attention hesaplaması
        Gerçek uygulamada CUDA optimizasyonlu natten kütüphanesi kullanılır
        """
        B, num_heads, _, _, head_dim = q.shape
        
        # Padding ekle
        pad = self.kernel_size // 2 * self.dilation
        
        # k ve v'yi padding ile genişlet
        k_padded = F.pad(
            k.permute(0, 1, 4, 2, 3),  # (B, heads, head_dim, H, W)
            (pad, pad, pad, pad),
            mode='constant',
            value=0
        ).permute(0, 1, 3, 4, 2)  # (B, heads, H+2*pad, W+2*pad, head_dim)
        
        v_padded = F.pad(
            v.permute(0, 1, 4, 2, 3),
            (pad, pad, pad, pad),
            mode='constant',
            value=0
        ).permute(0, 1, 3, 4, 2)
        
        # Her pozisyon için komşuluk attention hesapla
        output = torch.zeros_like(q)
        
        for i in range(H):
            for j in range(W):
                # Komşuluk bölgesini al
                i_start = i
                i_end = i + self.kernel_size
                j_start = j
                j_end = j + self.kernel_size
                
                # Dilation uygula
                k_neighborhood = k_padded[:, :, i_start:i_end:self.dilation, j_start:j_end:self.dilation, :]
                v_neighborhood = v_padded[:, :, i_start:i_end:self.dilation, j_start:j_end:self.dilation, :]
                
                # (B, heads, kernel_size, kernel_size, head_dim) -> (B, heads, kernel_size*kernel_size, head_dim)
                k_flat = k_neighborhood.reshape(B, num_heads, -1, head_dim)
                v_flat = v_neighborhood.reshape(B, num_heads, -1, head_dim)
                
                # Query for this position: (B, heads, 1, head_dim)
                q_pos = q[:, :, i, j, :].unsqueeze(2)
                
                # Attention scores: (B, heads, 1, k*k)
                attn = (q_pos @ k_flat.transpose(-2, -1)) * self.scale
                
                # Relative positional bias ekle
                rpb_idx = self._get_rpb_indices()
                attn = attn + self.rpb[:, rpb_idx[:, 0], rpb_idx[:, 1]].reshape(1, num_heads, 1, -1)
                
                attn = F.softmax(attn, dim=-1)
                attn = self.attn_drop(attn)
                
                # Output: (B, heads, 1, head_dim)
                out = attn @ v_flat
                output[:, :, i, j, :] = out.squeeze(2)
        
        # (B, heads, H, W, head_dim) -> (B, H, W, C)
        output = output.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        
        return output
    
    def _get_rpb_indices(self):
        """Relative positional bias indekslerini hesapla"""
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.kernel_size),
            torch.arange(self.kernel_size),
            indexing='ij'
        ))
        coords = coords.reshape(2, -1).T
        # Merkeze göre relatif pozisyon
        center = self.kernel_size // 2
        relative_coords = coords - center + (self.kernel_size - 1)
        return relative_coords


class NeighborhoodAttention2DEfficient(nn.Module):
    """
    Verimli Neighborhood Attention - Unfold kullanarak
    
    Bu versiyon, torch.unfold kullanarak daha verimli bir şekilde
    neighborhood attention hesaplar.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size // 2) * dilation
        
        # Q, K, V projeksiyonları
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Relative positional bias
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, kernel_size * kernel_size)
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) formatında giriş
        Returns:
            (B, H, W, C) formatında çıkış
        """
        B, H, W, C = x.shape
        
        # Q, K, V hesapla
        qkv = self.qkv(x)  # (B, H, W, 3*C)
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 5, 1, 2)  # (3, B, heads, head_dim, H, W)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Query'yi scale et
        q = q * self.scale
        
        # K ve V'yi unfold ile komşuluklara ayır
        k = F.pad(k, (self.padding,) * 4, mode='constant', value=0)
        v = F.pad(v, (self.padding,) * 4, mode='constant', value=0)
        
        # Unfold: (B, heads, head_dim, H, W) -> (B, heads, head_dim, H, W, k*k)
        k = k.unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
        v = v.unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
        
        # Reshape for attention
        k = k.reshape(B, self.num_heads, self.head_dim, H, W, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W, -1)
        
        # Attention: q @ k.T
        # q: (B, heads, head_dim, H, W) -> (B, heads, H, W, 1, head_dim)
        # k: (B, heads, head_dim, H, W, k*k) -> (B, heads, H, W, head_dim, k*k)
        q = q.permute(0, 1, 3, 4, 2).unsqueeze(-2)
        k = k.permute(0, 1, 3, 4, 2, 5)
        v = v.permute(0, 1, 3, 4, 2, 5)
        
        # Attention scores: (B, heads, H, W, 1, k*k)
        attn = q @ k
        
        # Positional bias ekle
        attn = attn + self.rpb.view(1, self.num_heads, 1, 1, 1, -1)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Output: (B, heads, H, W, 1, head_dim)
        out = attn @ v.transpose(-2, -1)
        
        # Reshape: (B, H, W, C)
        out = out.squeeze(-2).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        
        # Projeksiyon
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# =============================================================================
# NAT TRANSFORMER BLOKLARI
# =============================================================================

class Mlp(nn.Module):
    """MLP / Feed-Forward Network"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATBlock(nn.Module):
    """
    NAT Transformer Bloğu
    
    Her blok şunları içerir:
    1. LayerNorm
    2. Neighborhood Attention
    3. LayerNorm
    4. MLP (Feed-Forward Network)
    
    Residual bağlantılar her iki alt katmanda da kullanılır.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention2DEfficient(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) formatında giriş
        Returns:
            (B, H, W, C) formatında çıkış
        """
        # Attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic Depth - eğitim sırasında rastgele path'leri drop eder"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# =============================================================================
# PATCH EMBEDDING VE DOWNSAMPLING
# =============================================================================

class PatchEmbed(nn.Module):
    """
    Görüntüyü patch'lere böler ve embed eder
    
    Args:
        in_chans: Giriş kanal sayısı (RGB için 3)
        embed_dim: Embedding boyutu
        patch_size: Patch boyutu
    """
    
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 64,
        patch_size: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) formatında giriş
        Returns:
            (B, H//patch_size, W//patch_size, embed_dim) formatında çıkış
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging - çözünürlüğü yarıya indirip kanal sayısını 2x artırır
    
    Hiyerarşik yapı için kullanılır (Swin Transformer'a benzer)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) formatında giriş
        Returns:
            (B, H//2, W//2, 2*C) formatında çıkış
        """
        B, H, W, C = x.shape
        
        # Padding (eğer H veya W tek ise)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            B, H, W, C = x.shape
        
        # 2x2 komşu patch'leri birleştir
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


# =============================================================================
# NAT STAGE (BİRDEN FAZLA BLOK)
# =============================================================================

class NATStage(nn.Module):
    """
    NAT Stage - birden fazla NAT bloğu içerir
    
    Args:
        dim: Giriş boyutu
        depth: Bu stage'deki blok sayısı
        num_heads: Attention head sayısı
        kernel_size: Neighborhood boyutu
        dilations: Her blok için dilasyon değerleri
        downsample: Downsampling uygulansın mı
        mlp_ratio: MLP genişletme oranı
        qkv_bias: Q, K, V bias
        drop: Dropout oranı
        attn_drop: Attention dropout
        drop_path: DropPath oranları (liste)
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        kernel_size: int = 7,
        dilations: Optional[list] = None,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        
        # Dilasyon değerleri (varsayılan: hepsi 1)
        if dilations is None:
            dilations = [1] * depth
        
        # Drop path oranları
        if drop_path is None:
            drop_path = [0.0] * depth
        
        # NAT blokları
        self.blocks = nn.ModuleList([
            NATBlock(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilations[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
            )
            for i in range(depth)
        ])
        
        # Downsampling (opsiyonel)
        self.downsample = PatchMerging(dim) if downsample else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) formatında giriş
        Returns:
            Downsampling varsa (B, H/2, W/2, 2*C), yoksa (B, H, W, C)
        """
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


# =============================================================================
# TAM NAT MODELİ
# =============================================================================

class NAT(nn.Module):
    """
    Neighborhood Attention Transformer (NAT)
    
    Hiyerarşik vision transformer mimarisi.
    
    Args:
        img_size: Giriş görüntü boyutu
        patch_size: Patch boyutu
        in_chans: Giriş kanal sayısı
        num_classes: Sınıf sayısı
        embed_dim: İlk embedding boyutu
        depths: Her stage'deki blok sayıları
        num_heads: Her stage'deki head sayıları
        kernel_size: Neighborhood boyutu
        mlp_ratio: MLP genişletme oranı
        qkv_bias: Q, K, V bias
        drop_rate: Dropout oranı
        attn_drop_rate: Attention dropout
        drop_path_rate: Stochastic depth oranı
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 64,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [2, 4, 8, 16],
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim * (2 ** (self.num_stages - 1))
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # NAT Stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = NATStage(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                downsample=(i < self.num_stages - 1),  # Son stage'de downsample yok
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
            )
            self.stages.append(stage)
        
        # Classifier Head
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction"""
        x = self.patch_embed(x)  # (B, H', W', C)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)  # (B, H'', W'', C')
        x = x.permute(0, 3, 1, 2)  # (B, C', H'', W'')
        x = x.flatten(2)  # (B, C', H''*W'')
        x = self.avgpool(x)  # (B, C', 1)
        x = x.flatten(1)  # (B, C')
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) formatında giriş görüntüsü
        Returns:
            (B, num_classes) formatında sınıf logitleri
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


# =============================================================================
# ÖNCEDEn TANIMLI NAT MODELLERİ
# =============================================================================

def nat_mini(num_classes: int = 2, **kwargs) -> NAT:
    """NAT-Mini: Küçük ve hızlı model"""
    return NAT(
        embed_dim=64,
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        kernel_size=7,
        num_classes=num_classes,
        **kwargs
    )


def nat_tiny(num_classes: int = 2, **kwargs) -> NAT:
    """NAT-Tiny: Orta boy model"""
    return NAT(
        embed_dim=64,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        kernel_size=7,
        num_classes=num_classes,
        **kwargs
    )


def nat_small(num_classes: int = 2, **kwargs) -> NAT:
    """NAT-Small: Daha büyük model"""
    return NAT(
        embed_dim=64,
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        kernel_size=7,
        num_classes=num_classes,
        **kwargs
    )


def nat_base(num_classes: int = 2, **kwargs) -> NAT:
    """NAT-Base: En büyük model"""
    return NAT(
        embed_dim=128,
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        kernel_size=7,
        num_classes=num_classes,
        **kwargs
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test NAT model
    print("NAT Model Test")
    print("=" * 50)
    
    # Örnek giriş
    batch_size = 2
    img_size = 224
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Model oluştur
    model = nat_mini(num_classes=2)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Model parametreleri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ NAT model test passed!")


"""
BreaKHis Dataset Downloader
===========================

Bu script Kaggle'dan BreaKHis veri setini otomatik olarak indirir.

Kullanım:
    python dataset.py

Gereksinimler:
    pip install kagglehub

Not: Kaggle API key'inizi ayarlamanız gerekebilir:
    - https://www.kaggle.com/settings adresinden API token indirin
    - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
    - Linux: ~/.kaggle/kaggle.json
"""

import os
import sys
from pathlib import Path


def download_breakhis_dataset():
    """BreaKHis veri setini indir"""
    
    print("=" * 60)
    print(" BreaKHis Dataset Downloader ")
    print("=" * 60)
    
    try:
        import kagglehub
    except ImportError:
        print("\n❌ kagglehub kütüphanesi bulunamadı!")
        print("   Lütfen şu komutu çalıştırın: pip install kagglehub")
        sys.exit(1)
    
    print("\n📥 Downloading BreaKHis dataset from Kaggle...")
    print("   (Bu işlem birkaç dakika sürebilir)\n")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("ambarish/breakhis")
        
        print("\n" + "=" * 60)
        print("✅ Dataset successfully downloaded!")
        print("=" * 60)
        print(f"\n📁 Path to dataset files: {path}")
        
        # Dosya yapısını kontrol et
        dataset_path = Path(path)
        if dataset_path.exists():
            print("\n📂 Dataset structure:")
            for item in sorted(dataset_path.iterdir())[:10]:
                print(f"   - {item.name}")
            
            # PNG dosyalarını say
            png_count = len(list(dataset_path.rglob("*.png")))
            print(f"\n📊 Total PNG images found: {png_count}")
        
        return path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Kaggle API key ayarlı mı kontrol edin")
        print("   2. İnternet bağlantınızı kontrol edin")
        print("   3. kagglehub versiyonunu güncelleyin: pip install --upgrade kagglehub")
        sys.exit(1)


def check_dataset_exists():
    """Dataset'in mevcut olup olmadığını kontrol et"""
    
    import platform
    
    if platform.system() == "Windows":
        base = Path(os.path.expanduser(r"~\.cache\kagglehub\datasets\ambarish\breakhis"))
    else:
        base = Path(os.path.expanduser("~/.cache/kagglehub/datasets/ambarish/breakhis"))
    
    if base.exists():
        versions = sorted(base.glob("versions/*"), key=lambda x: int(x.name) if x.name.isdigit() else 0)
        if versions:
            latest = versions[-1]
            breast_path = latest / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
            if breast_path.exists():
                png_count = len(list(breast_path.rglob("*.png")))
                if png_count > 0:
                    return True, str(breast_path), png_count
    
    return False, None, 0


if __name__ == "__main__":
    # Önce mevcut veri setini kontrol et
    exists, path, count = check_dataset_exists()
    
    if exists:
        print("=" * 60)
        print(" BreaKHis Dataset Status ")
        print("=" * 60)
        print(f"\n✅ Dataset already exists!")
        print(f"📁 Path: {path}")
        print(f"📊 Images: {count}")
        
        user_input = input("\n🔄 Re-download? (y/N): ").strip().lower()
        if user_input != 'y':
            print("\n👍 Using existing dataset.")
            sys.exit(0)
    
    # İndir
    download_breakhis_dataset()

import os
import sys
from typing import Optional
from qdrant_client import QdrantClient, models  # ✅ Doğru import yapısı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager  


# 🔑 API anahtarlarını yükle
key_manager = KeyManager()
api_keys = key_manager.load_api_key()
qdrant_host = api_keys.get('qdrant_host', '')
qdrant_api_key = api_keys.get('qdrant_api_key', '')


class QdrantDatabase:
    """Qdrant veritabanı bağlantısı için yardımcı sınıf"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        self._connected = False  # ✅ Bağlantı durumu kontrolü

    def connect(self):
        """Qdrant sunucusuna bağlanır"""
        if not self.client:
            try:
                self.client = QdrantClient(
                    url=qdrant_host,
                    api_key=qdrant_api_key,
                    timeout=300,
                    verify=False  # ⚠️ SSL is correct or not
                    )

                # Bağlantıyı test et
                self.client.get_collections()
                if not self._connected:
                    print("✅ Qdrant bağlantısı başarılı!")
                    self._connected = True

            except Exception as e:
                print(f"❌ Qdrant bağlantı hatası: {str(e)}")
                raise

    def _ensure_client(self):
        """Client'ın bağlı olduğundan emin ol"""
        if not self.client:
            self.connect()

    def get_client(self) -> QdrantClient:
        """Qdrant client'ı döndürür"""
        self._ensure_client()
        return self.client

    def get_models(self):
        """Qdrant modellerine erişim sağlar"""
        return models  # ✅ Artık doğrudan import edildi

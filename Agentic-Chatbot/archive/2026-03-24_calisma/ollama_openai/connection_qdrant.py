import os
import sys
from typing import Optional
from qdrant_client import AsyncQdrantClient, models  # ✅ Doğru import yapısı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager  


# 🔑 API anahtarlarını yükle
key_manager = KeyManager()
api_keys = key_manager.load_api_key()
qdrant_host = api_keys.get('qdrant_host', '')
qdrant_api_key = api_keys.get('qdrant_api_key', '')


class QdrantDatabase:
    """Qdrant veritabanı bağlantısı için yardımcı sınıf (Async)"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name
        self.client: Optional[AsyncQdrantClient] = None
        self._connected = False  # ✅ Bağlantı durumu kontrolü

    def connect(self):
        """Qdrant client'ı oluşturur (async bağlantı lazy olarak yapılır)"""
        if not self.client:
            try:
                self.client = AsyncQdrantClient(
                    url=qdrant_host,
                    api_key=qdrant_api_key,
                    timeout=300,
                    check_compatibility=False
                )
                print("✅ Qdrant AsyncClient oluşturuldu!")
                self._connected = True

            except Exception as e:
                print(f"❌ Qdrant client oluşturma hatası: {str(e)}")
                raise

    async def async_connect(self):
        """Async bağlantı testi yapar"""
        if not self.client:
            self.connect()
        try:
            # Async bağlantı testi
            await self.client.get_collections()
            print("✅ Qdrant async bağlantısı başarılı!")
            return True
        except Exception as e:
            print(f"❌ Qdrant bağlantı hatası: {str(e)}")
            return False

    def _ensure_client(self):
        """Client'ın oluşturulduğundan emin ol"""
        if not self.client:
            self.connect()

    def get_client(self) -> AsyncQdrantClient:
        """Qdrant client'ı döndürür"""
        self._ensure_client()
        return self.client

    def get_models(self):
        """Qdrant modellerine erişim sağlar"""
        return models  # ✅ Artık doğrudan import edildi

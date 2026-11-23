import json
from typing import List, Dict, Any, Optional
from db_key.db import KeyManager

key_manager = KeyManager()
api_keys = key_manager.load_api_key()
openai_api_key = api_keys.get('gpt_api_key', '')

from openai import OpenAI

client = OpenAI(api_key=openai_api_key)
from agent.simpleAgent import SimpleAgent
from prompts.handle_db import PromptManager

class AgentManager:
    def __init__(self):
        self.question_analyzer_agent = None
        self.reply_agent = None
        self.query_router_agent = None
        self.follow_up_agent = None
        self.response_validator_agent = None
        # Yeni agent'lar eklendi
        self.qdrant_search_agent = None
        self.agent_map_agent = None
        self.chat_history_agent = None
        self.tour_detail_agent = None
        self.mcp_service_agent = None
        self.greeting_agent = None
        self._initialized = False

    async def initialize_agents(self, prompt=None):
        print("🔧 Agent'lar initialize ediliyor...")
        
        # Eğer prompt verilmemişse varsayılan prompt'u yükle
        if prompt is None:
            print("📝 Prompt yükleniyor...")
            prompt_manager = PromptManager()
            await prompt_manager.load_prompts("pro")  # Varsayılan olarak pro kullan
            prompt = prompt_manager.prompts
            print(f"✅ Prompt yüklendi: {list(prompt.keys())}")
        
        print(f"📋 Kullanılacak prompt anahtarları: {list(prompt.keys())}")
        
        # Agent'ları oluştur
        try:
            # Question Analyzer Agent
            self.question_analyzer_agent = SimpleAgent(
                name="Question Analyzer",
                model="gpt-4.1",
                instructions=prompt.get("question_analyzer_system_prompt", "Soruyu analiz et")
            )
            print("✅ Question Analyzer Agent oluşturuldu")

            # Reply Agent
            self.reply_agent = SimpleAgent(
                name="Reply Agent",
                model="gpt-4.1",
                instructions=prompt.get("repl_agent_system_prompt", "Yanıt ver")
            )
            print("✅ Reply Agent oluşturuldu")

            # Query Router Agent
            self.query_router_agent = SimpleAgent(
                name="Query Router",
                model="gpt-4.1",
                instructions=prompt.get("query_router_agent_system_prompt", "Sorguyu yönlendir")
            )
            print("✅ Query Router Agent oluşturuldu")

            # Follow Up Agent
            self.follow_up_agent = SimpleAgent(
                name="Follow Up Agent",
                model="gpt-4.1",
                instructions=prompt.get("follow_up_agent_system_prompt", "Takip soruları sor")
            )   
            print("✅ Follow Up Agent oluşturuldu")

            # Response Validator Agent
            self.response_validator_agent = SimpleAgent(
                name="Response Validator",
                model="gpt-4.1",
                instructions=prompt.get("response_validator_agent_system_prompt", "Yanıtı doğrula")
            )
            print("✅ Response Validator Agent oluşturuldu")

            # Qdrant Search Agent
            self.qdrant_search_agent = SimpleAgent(
                name="Qdrant Search",
                model="gpt-4.1",
                instructions=prompt.get("qdrant_search_system_prompt", "Qdrant araması yap")
            )
            print("✅ Qdrant Search Agent oluşturuldu")

            # Agent Map Agent
            self.agent_map_agent = SimpleAgent(
                name="Agent Map",
                model="gpt-4.1",
                instructions=prompt.get("agent_map_system_prompt", "Tur listelerini filtrele")
            )
            print("✅ Agent Map Agent oluşturuldu")

            # Chat History Agent
            self.chat_history_agent = SimpleAgent(
                name="Chat History",
                model="gpt-4.1",
                instructions=prompt.get("chat_history_system_prompt", "Konuşma geçmişini yönet")
            )
            print("✅ Chat History Agent oluşturuldu")

            # Tour Detail Agent
            self.tour_detail_agent = SimpleAgent(
                name="Tour Detail",
                model="gpt-4.1",
                instructions=prompt.get("determine_agent_and_extract_system_prompt", "Tur detaylarını çıkar")
            )
            print("✅ Tour Detail Agent oluşturuldu")

            # MCP Service Agent
            self.mcp_service_agent = SimpleAgent(
                name="MCP Service",
                model="gpt-4.1",
                instructions=prompt.get("mcp_service_agent_system_prompt", "MCP servisini çağır")
            )
            print("✅ MCP Service Agent oluşturuldu")

            # Greeting Agent
            self.greeting_agent = SimpleAgent(
                name="Greeting",
                model="gpt-4.1",
                instructions=prompt.get("greeting_agent_system_prompt", "Selamlaşma yanıtları ver")
            )
            print("✅ Greeting Agent oluşturuldu")
            
            self._initialized = True
            print("🎉 Tüm agent'lar başarıyla oluşturuldu!")
            
        except Exception as e:
            print(f"❌ Agent oluşturma hatası: {str(e)}")
            raise e
        
        return {
            "question_analyzer_agent": self.question_analyzer_agent,
            "reply_agent": self.reply_agent,
            "query_router_agent": self.query_router_agent,
            "follow_up_agent": self.follow_up_agent,
            "response_validator_agent": self.response_validator_agent,
            "qdrant_search_agent": self.qdrant_search_agent,
            "agent_map_agent": self.agent_map_agent,
            "chat_history_agent": self.chat_history_agent,
            "tour_detail_agent": self.tour_detail_agent,
            "mcp_service_agent": self.mcp_service_agent,
            "greeting_agent": self.greeting_agent
        }

    def is_initialized(self):
        return self._initialized

# Global agent manager instance
agent_manager = AgentManager()

# Backward compatibility için eski fonksiyon
async def initialize_agents(prompt=None):
    return await agent_manager.initialize_agents(prompt)

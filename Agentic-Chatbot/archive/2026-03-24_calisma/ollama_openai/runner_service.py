import os
import sys
import json
import httpx
from typing import Optional, List, Type
from pydantic import BaseModel
import token_price as price
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager  # <-- Python 3.9 için gerekli

key_manager = KeyManager()
api_list = key_manager.load_api_key()
llm_api_key = api_list.get('ollama_api_key', '')
system_room_base_url = api_list.get('system_room_base_url', '')
model_llm = api_list.get('model_ollama', '')

class AgentTool:
    """
    Bir agent'ı başka bir agent için tool haline getirir.
    """
    def __init__(self, name: str, description: str, agent: "Agent"):
        self.name = name
        self.description = description
        self.agent = agent

    def call(self, input_text: str) -> str:
        return self.agent._run_with_llm(input_text)

class Agent:
    """
    Tek bir görevi yerine getiren agent sınıfı.
    Her agent'ın bir adı, talimatları ve opsiyonel tool'ları vardır.
    """
    def __init__(
        self,
        name: str,
        instructions: str,
        model_llm: str = model_llm,
        use_llm: bool = True,
        llm_base_url: str = None,
        tools: Optional[List[AgentTool]] = None,
        output_type: Optional[Type[BaseModel]] = None,   # 🔥 yeni parametre
    ):
        
        self.name = name
        self.instructions = instructions.strip()
        self.model_llm = model_llm
        self.use_llm = use_llm
        self.llm_base_url = llm_base_url or f"http://{system_room_base_url}:8007"
        self.tools = tools or []
        self.output_type = output_type
        
    # -------------------------
    # Tool'a dönüştürme
    # -------------------------
    def as_tool(self, tool_name: str, tool_description: str) -> AgentTool:
        return AgentTool(
            name=tool_name,
            description=tool_description,
            agent=self,
        )

    # -------------------------
    # Tool çağrısı kontrolü
    # -------------------------
    def _maybe_call_tool(self, input_text: str) -> Optional[str]:
        """
        Basit routing: input içinde tool ismi geçiyorsa ilgili agent çağrılır.
        Daha gelişmiş yapıda LLM tool selection yapılabilir.
        """
        for tool in self.tools:
            if tool.name in input_text.lower():
                return tool.call(input_text)
        return None
    
    

    async def run(self, input_text: str, chat_history: list = None):
        """
        Agent'ı çalıştır.

        Args:
            input_text:    Son kullanıcı mesajı / sorgusu
            chat_history:  OpenAI messages formatında geçmiş
                        [{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]
        """
        if not self.use_llm:
            return {"error": "Vllm kullanımına izin verilmedi."}

        raw_result = await self._run_with_llm(input_text, chat_history)
        raw_str = (raw_result or "").strip()

        # -------------------------------
        # Token hesaplama ve Usage objesi
        # -------------------------------
        history_text = ""
        if chat_history:
            history_text = "\n".join(m.get("content", "") for m in chat_history)
        input_tokens = price.count_tokens(self.instructions + "\n" + history_text + "\n" + input_text)
        output_tokens = price.count_tokens(raw_str)
        total_tokens = input_tokens + output_tokens

        usage = price.Usage(
            requests=1,
            input_tokens=input_tokens,
            input_tokens_details=price.InputTokensDetails(cached_tokens=input_tokens),
            output_tokens=output_tokens,
            output_tokens_details=price.OutputTokensDetails(reasoning_tokens=output_tokens),
            total_tokens=total_tokens
        )

        # -------------------------------
        # JSON parse denemesi
        # -------------------------------
        # [#ID] TOON formatı veya [text] gibi kalıpları JSON sanmayı önle
        looks_like_json = (
            raw_str.startswith("{")
            or (raw_str.startswith("[") and not raw_str.startswith("[#"))
        )
        if looks_like_json:
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, dict):
                    parsed["usage"] = usage
                    return parsed
                return {"types": ["text"], "text": str(parsed), "usage": usage}
            except json.JSONDecodeError as e:
                display = (raw_str[:1000] + '...') if len(raw_str) > 1000 else raw_str
                print(f"⚠️ [Agent: {self.name}] JSON parse hatası: {e}. Raw (truncated): {display}")

        # -------------------------------
        # Normal text dönüşü
        # -------------------------------
        return {
            "types": ["text"],
            "text": raw_str,
            "usage": usage,
            "token_count": total_tokens,
            "token_cost": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_type": "llm"
        }

    async def _run_with_llm(self, input_text: str, chat_history: list = None) -> str:
        try:
            # -------------------------------------------------------
            # Endpoint belirleme:
            #   - Model adı "openai/" ile başlıyorsa → OpenAI-compat API  (/v1/chat/completions)
            #   - Diğer durumlarda → Ollama native API  (/api/chat)
            # -------------------------------------------------------
            is_openai_compat = self.model_llm.startswith("openai/")

            if is_openai_compat:
                url = f"{self.llm_base_url}/v1/chat/completions"
            else:
                url = f"{self.llm_base_url}/api/chat"

            # -------------------------------------------------------
            # Messages dizisi oluştur:
            #   [system] → agent talimatları
            #   [user/assistant ...] → chat geçmişi (varsa)
            #   [user] → son kullanıcı mesajı
            # -------------------------------------------------------
            messages = [{"role": "system", "content": self.instructions}]
            if chat_history:
                messages.extend(chat_history)
            messages.append({"role": "user", "content": input_text})

            payload = {
                "model": self.model_llm,
                "messages": messages,
                "temperature": 0.9,
            }

            # Ollama native → stream: false gerekli
            if not is_openai_compat:
                payload["stream"] = False

            headers = {
                "Content-Type": "application/json",
            }
            if llm_api_key:
                headers["Authorization"] = f"Bearer {llm_api_key}"

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    return json.dumps({
                        "error": f"LLM API hatası: {response.status_code}",
                        "detail": response.text
                    })

                result = response.json()

                # -------------------------------------------------------
                # Response parsing:
                #   OpenAI format : result["choices"][0]["message"]["content"]
                #   Ollama format : result["message"]["content"]
                # -------------------------------------------------------
                if is_openai_compat:
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
                    return ""
                else:
                    return result.get("message", {}).get("content", "").strip()

        except httpx.RequestError as e:
            return json.dumps({"error": f"LLM bağlantı hatası: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"LLM çalıştırma hatası: {str(e)}"})

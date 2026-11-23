import os
# region Kütüphaneler
from utils import system_prompt as custom_rules,determine_agent_and_extract_system_prompt,repl_agent_system_prompt,query_router_agent_system_prompt,follow_up_agent_system_prompt,question_analyzer_agent_system_prompt,response_validator_agent_system_prompt
from typing import Dict, Any, List, Optional, Union, cast
import json,os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(api_key=openai_api_key)

# endregion


class SimpleAgent:
    def __init__(self, name: str, model: str, instructions: str):
        self.name = name
        self.model = model
        self.instructions = instructions

    async def run(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None, input_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        try:
            # Context'i mesajlara ekle
            if context:
                # Set'leri list'e dönüştür
                if isinstance(context, dict):
                    for key, value in context.items():
                        if isinstance(value, set):
                            context[key] = list(value)
                context_message = f"Context: {json.dumps(context, ensure_ascii=False)}"
                messages.append({"role": "system", "content": context_message})
            
            # input_data'yı mesajlara ekle
            if input_data:
                input_data_message = f"input_data: {json.dumps(input_data, ensure_ascii=False)}"
                messages.append({"role": "system", "content": input_data_message})
            
            # Agent talimatlarını ekle
            messages.append({
                "role": "system", 
                "content": f"{self.instructions}\n\nLütfen cevabınızı JSON formatında verin. Örnek format: {{\"answer\": \"cevap\", \"source\": \"kaynak\", \"confidence\": 0.95}}"
            })
            
            # OpenAI'ya istek at
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                response_format={"type": "json_object"}
            )
            
            # JSON cevabını parse et
            content = response.choices[0].message.content
            if not content:
                return {"answer": "Cevap alınamadı", "source": "gpt", "confidence": 0.0}
            
            try:
                result = json.loads(content)
                # Eğer answer anahtarı yoksa, tüm içeriği answer olarak kullan
                if "answer" not in result:
                    result = {"answer": content, "source": "gpt", "confidence": 0.8}
                return result
            except json.JSONDecodeError:
                return {"answer": content, "source": "gpt", "confidence": 0.8}
                
        except Exception as e:
            print(f"Agent hatası: {str(e)}")
            return {"answer": f"Bir hata oluştu: {str(e)}", "source": "error", "confidence": 0.0}


# Agent'ları oluştur
question_analyzer_agent = SimpleAgent(
    name="Question Analyzer",
    model="gpt-4.1",
    instructions=question_analyzer_agent_system_prompt
)

# Reply agent'ı oluştur
reply_agent = SimpleAgent(
    name="Reply Agent",
    model="gpt-4.1",
    instructions=repl_agent_system_prompt
)

query_router_agent = SimpleAgent(
    name="Query Router",
    model="gpt-4.1",
    instructions=query_router_agent_system_prompt
)

follow_up_agent = SimpleAgent(
    name="Follow Up Agent",
    model="gpt-4.1",
    instructions=follow_up_agent_system_prompt
)

# Response validator agent'ı oluştur
response_validator_agent = SimpleAgent(
    name="Response Validator",
    model="gpt-4.1",
    instructions=response_validator_agent_system_prompt
)

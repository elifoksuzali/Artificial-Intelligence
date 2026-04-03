
import threading
import asyncio
import json
import os
import warnings



import openai
import os
import requests
from typing import Dict, Any
from datetime import datetime



ollama_base_url="ollama_base_url"
model="gpt-oss:20b"
use_ollama=True,  # Ollama kullanımını aktif et


class Agent:
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o-mini", use_ollama: bool = False, ollama_base_url: str = ollama_base_url,tools:list=[]):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        self.tools = tools or []
    def run(self, input_text: str) -> str:
        try:
            if self.use_ollama:
                return self._run_with_ollama(input_text)
            else:
                return self._run_with_openai(input_text)
        except Exception as e:
            return f"Agent hatası: {str(e)}"
    
    def _run_with_ollama(self, input_text: str) -> str:
        """
        Ollama API kullanarak model çalıştırır
        """
        try:
            # Ollama API endpoint - /api/chat kullanılıyor
            url = f"{ollama_base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": input_text}
                ],
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                return f"Ollama API hatası: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Ollama çalıştırma hatası: {str(e)}"
    
    def _run_with_openai(self, input_text: str) -> str:
        """
        OpenAI API kullanarak model çalıştırır
        """
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI çalıştırma hatası: {str(e)}"

class Runner:
    @staticmethod
    def run_sync(agent: Agent, input_text: str) -> Any:
        """
        Agent'ı senkron olarak çalıştırır
        """
        try:
            result = agent.run(input_text)
            
            # Result objesi oluştur
            class Result:
                def __init__(self, final_output: str):
                    self.final_output = final_output
                    self.timestamp = datetime.now().isoformat()
            
            return Result(result)
            
        except Exception as e:
            # Hata durumunda da Result objesi döndür
            class Result:
                def __init__(self, error: str):
                    self.final_output = f"Hata: {error}"
                    self.timestamp = datetime.now().isoformat()
            
            return Result(str(e)) 

# RuntimeWarning'leri bastır
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")



# ---------------- Greeting Agent ----------------
greeting_agent = Agent(
    name="GreetingAgent",
    instructions="""
    You are a friendly greeting agent.
    Respond to any greeting in a polite and cheerful way.
    Always include the user's name and title if available.
    """,
    model=model,
    use_ollama=use_ollama,
    ollama_base_url=ollama_base_url
)


def call_greeting_agent(message: str) -> str:
    return Runner.run_sync(greeting_agent, message).final_output

# ---------------- Analysis Agent ----------------
analysis_agent = Agent(
    name="AnalysisAgent",
    instructions="""
    You are a message analysis agent.
    Analyze incoming messages and determine if it is a greeting or not.
    
    You must respond with a JSON object in this exact format:
    - If it's a greeting: {"label":"greeting"}
    - If it's not a greeting: {"label":"other"}
    
    Do not call any tools directly. Just analyze and return the JSON.
    """,
    tools=[],
    model=model,
    use_ollama=use_ollama,
    ollama_base_url=ollama_base_url
)

# ---------------- Main Processing Agent ----------------
main_agent = Agent(
    name="MainAgent",
    instructions="""
    You are the main processing agent.
    You will receive analysis results and decide what to do.
    If the analysis shows {"label":"greeting"}, call the greeting tool.
    Otherwise, respond normally to the message.
    """,
    tools=[call_greeting_agent],
    model=model,
    use_ollama=use_ollama,
    ollama_base_url=ollama_base_url
)

# ---------------- Message Processing ----------------
def process_message_sync(message: str):
    """Sync message processing with proper event loop handling"""
    try:
        print(f"=== Processing message: {message} ===")
        
        # 1. Analysis agent ile mesajı analiz et
        print("[Step 1] Analyzing message...")
        analysis_result = Runner.run_sync(analysis_agent, input_text=message)
        analysis_output = analysis_result.final_output
        print(f"[Event] Analysis output: {analysis_output}")
        
        # 2. Analysis sonucunu kontrol et
        try:
            analysis_data = json.loads(analysis_output)
            print(f"[Step 2] Parsed analysis: {analysis_data}")
            
            if analysis_data.get("label") == "greeting":
                print("[Step 3] Detected greeting - calling greeting tool...")
                # Greeting tool'unu çağır
                greeting_result = Runner.run_sync(greeting_agent, input_text=message)
                print(f"[Event] Greeting tool result: {greeting_result.final_output}")
                final_output = greeting_result.final_output
            else:
                print("[Step 3] Not a greeting - responding normally...")
                final_output = f"Normal response for: {message}"
                
        except json.JSONDecodeError:
            print("[Error] Could not parse analysis output as JSON")
            final_output = f"Error analyzing: {message}"
        
        return final_output
        
    except Exception as e:
        print(f"Error processing {message}: {str(e)}")
        raise e

# ---------------- Thread Worker ----------------
class MessageWorker(threading.Thread):
    def __init__(self, message: str, on_complete=None):
        super().__init__(daemon=True)
        self.message = message
        self.on_complete = on_complete
        self.result = None
        self.error = None

    def run(self):
        # Her thread için tamamen izole edilmiş event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Sync fonksiyonu çalıştır
            self.result = process_message_sync(self.message)
        except Exception as e:
            self.error = str(e)
            print(f"Error processing {self.message}: {str(e)}")
        finally:
            # Tüm pending taskları temizle
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass  # Ignore cleanup errors
            
            # Event loop'u temizle
            try:
                loop.close()
            except Exception:
                pass  # Ignore cleanup errors
            
            asyncio.set_event_loop(None)
            
            if self.on_complete:
                self.on_complete(self)

# ---------------- Örnek Kullanım ----------------
if __name__ == "__main__":
    messages = [
        "Merhaba, nasılsınız?"
    ]

    active_workers = []

    def remove_worker(worker):
        active_workers.remove(worker)

    # Thread başlat
    for msg in messages:
        w = MessageWorker(msg, on_complete=remove_worker)
        active_workers.append(w)
        w.start()

    # Tüm threadlerin bitmesini bekle
    for w in active_workers[:]:
        w.join()


#============================ OPENAI MİMARİ YAPISI =========================


import os
import json
import time
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# =========================================
# CONFIG
# =========================================

MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================
# OpenAI Safe Call with Retry
# =========================================

def call_openai_with_retry(messages: List[Dict[str, str]], model: str = MODEL_NAME) -> str:
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                timeout=REQUEST_TIMEOUT,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()

            # Retry only transient errors
            if any(code in error_str for code in ["429", "500", "502", "503", "timeout"]):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            # Critical error → stop
            raise e

    return "Temporary service issue. Please try again."

# =========================================
# Agent Class
# =========================================

class Agent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions

    def run(self, messages: List[Dict[str, str]]) -> str:

        final_messages = [
            {"role": "system", "content": self.instructions},
            *messages
        ]

        return call_openai_with_retry(final_messages)

# =========================================
# Agents
# =========================================

analysis_agent = Agent(
    name="AnalysisAgent",
    instructions="""
You are a message classification agent.

Return ONLY a JSON in this exact format:
{"label":"greeting"}
or
{"label":"other"}

Do not write anything else.
"""
)

greeting_agent = Agent(
    name="GreetingAgent",
    instructions="""
You are a friendly greeting assistant.
Respond politely and cheerfully.
If user's name exists in previous messages, use it.
"""
)

main_agent = Agent(
    name="MainAgent",
    instructions="""
You are a professional AI assistant.
Respond clearly and helpfully.
"""
)

# =========================================
# Router Logic
# =========================================

def process_message(previous_messages: List[Dict[str, str]], new_message: str) -> str:

    full_context = previous_messages + [
        {"role": "user", "content": new_message}
    ]

    # Step 1 → Analyze
    analysis_output = analysis_agent.run([
        {"role": "user", "content": new_message}
    ])

    try:
        analysis_data = json.loads(analysis_output)

        if analysis_data.get("label") == "greeting":
            return greeting_agent.run(full_context)
        else:
            return main_agent.run(full_context)

    except json.JSONDecodeError:
        # Fallback güvenliği
        return main_agent.run(full_context)

# =========================================
# FastAPI Layer
# =========================================

app = FastAPI()

class ChatRequest(BaseModel):
    previous_messages: List[Dict[str, str]]
    message: str


@app.post("/chat")
def chat_endpoint(request: ChatRequest):

    try:
        response_text = process_message(
            previous_messages=request.previous_messages,
            new_message=request.message
        )

        return {
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
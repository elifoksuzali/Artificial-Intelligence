import openai
import os
import requests
from typing import Dict, Any
from datetime import datetime
from config import OLLAMA_BASE_URL
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from agents import Runner, Agent
class Agent:
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o-mini", use_ollama: bool = False, ollama_base_url: str = OLLAMA_BASE_URL):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        
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
            url = f"{self.ollama_base_url}/api/chat"
            
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


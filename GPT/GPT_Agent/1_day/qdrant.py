# orchestrator_with_qdrant.py

import openai
from typing import List, Dict
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.qdrantDb import QdrantDatabase
# --- Qdrant Context Store ---
class QdrantContextStore:
    def __init__(self):
        self.context: List[Dict] = []

    def get_context(self):
        return self.context

    def update_context(self, new_items: List[Dict]):
        self.context.extend(new_items)


# --- Qdrant Fetch Agent ---
class FetchAgent:
    def __init__(self):
        self.qdrant = QdrantDatabase("mng-cosine")

    def fetch(self, user_input: str):
        # Örnek: OpenAI embedding ile vektör çıkar
        embedding = self.get_embedding(user_input)
        search_results = self.qdrant.search(embedding, limit=5)

        return [
            {"text": item.payload.get("text") or str(item.payload)}
            for item in search_results if item.payload
        ]

    def get_embedding(self, text: str):
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


# --- Answer Agent ---
class AnswerAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def answer(self, user_input: str, context: List[Dict]) -> str:
        context_text = "\n".join([item['text'] for item in context])
        system_prompt = f"""
        Aşağıdaki bilgilere dayanarak kullanıcının sorusunu yanıtla:
        Kullanıcı sorusu: {user_input}
        Veriler:\n{context_text}
        Cevap kısa, doğru ve kullanıcı dostu olsun.
        """
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt}
            ]
        )
        return completion.choices[0].message.content


# --- Orchestrator Agent ---
class OrchestratorAgent:
    def __init__(self):
        self.context_store = QdrantContextStore()
        self.fetch_agent = FetchAgent()
        self.answer_agent = AnswerAgent()

    def handle_user_input(self, user_input: str) -> str:
        context = self.context_store.get_context()

        if self._context_has_answer(user_input, context):
            print("📦 Mevcut context yeterli. Yanıt oluşturuluyor...")
            return self.answer_agent.answer(user_input, context)
        else:
            print("📥 Yetersiz context. Yeni veri çekiliyor...")
            new_context = self.fetch_agent.fetch(user_input)
            self.context_store.update_context(new_context)
            updated_context = self.context_store.get_context()
            return self.answer_agent.answer(user_input, updated_context)

    def _context_has_answer(self, user_input: str, context: List[Dict]) -> bool:
        user_keywords = set(user_input.lower().split())
        for item in context:
            if user_keywords.intersection(item["text"].lower().split()):
                return True
        return False


# --- Test CLI ---
if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    while True:
        question = input("\n❓ Sorunuz: ")
        if question.lower() in ["exit", "çık", "q"]:
            break
        response = orchestrator.handle_user_input(question)
        print(f"🤖 Yanıt: {response}")

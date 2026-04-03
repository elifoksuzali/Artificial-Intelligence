

import streamlit as st
import openai
import time
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.qdrantDb import QdrantDatabase
from context.get_context import get_context
from prompting.chat_utils import system_prompt as custom_rules
import agent.agent

# OpenAI API key
openai.api_key = "openai.api_key"
ASSISTANT_ID = "ASSISTANT_ID"  # <- senin assistant ID'in

# Setup Streamlit
st.set_page_config(page_title="Asistan", layout="wide")
st.title("💬 GPT Asistanı")

# Yazma animasyonu
def typing_effect(text, speed=0.02):
    output = ""
    placeholder = st.empty()
    for char in text:
        output += char
        placeholder.markdown(f"<div style='font-size: 18px;'>{output}</div>", unsafe_allow_html=True)
        time.sleep(speed)

# Konuşma geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id

# Bubble stilinde geçmiş mesajları göster
def chat_bubble(role, message, is_user=False):
    if is_user:
        return f"""
        <div style='text-align: right; margin: 8px 0;'>
            <span style='background-color: green; color: white; padding: 10px 15px; border-radius: 20px; display: inline-block; max-width: 70%;'>
                {message}
            </span>
        </div>
        """
    elif role == "assistant":
        return f"""
        <div style='text-align: left; margin: 8px 0;'>
            <span style='background-color: darkorange; color: white; padding: 10px 15px; border-radius: 20px; display: inline-block; max-width: 70%;'>
                {message}
            </span>
        </div>
        """

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(chat_bubble("user", msg["content"], is_user=True), unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(chat_bubble("assistant", msg["content"]), unsafe_allow_html=True)

# Kullanıcıdan yeni mesaj
st.markdown("---")
soru = st.chat_input("Mesajınızı yazın...")

if soru:
    # Mesajı geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": soru})

    # 🔍 Qdrant'tan bağlam çek
    qdrant_context = get_context(soru, q_client=QdrantDatabase("mng-cosine"), num_results=10)
    context_text = "\n".join([f"- {c}" for c in qdrant_context])

    # 🧠 Qdrant bağlamı ve soru birleştirilerek Assistant'a gönderiliyor
    prompt_text = f"""## CONTEXT
{context_text}

## USER MESSAGE
{soru}

## SYSTEM INSTRUCTIONS
{custom_rules}
"""

    with st.spinner("GPT düşünüyor..."):

        # 📨 Mesajı Assistant thread’ine gönder
        openai.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt_text
        )

        # 🚀 Run başlat
        run = openai.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=ASSISTANT_ID
        )

        # ⏳ Run tamamlanana kadar bekle
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["queued", "in_progress"]:
                time.sleep(1)
            else:
                st.error("⚠️ Assistant çalıştırma başarısız oldu.")
                break

        # ✅ Assistant yanıtını al
        messages = openai.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        assistant_message = messages.data[0].content[0].text.value.strip()

        # Geçmişe kaydet ve göster
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        typing_effect(assistant_message)

    # Sayfa yenile
    st.rerun()

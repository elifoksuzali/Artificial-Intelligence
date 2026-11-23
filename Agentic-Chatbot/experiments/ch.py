import os
import openai
import time
import json
openai.api_key = os.getenv("OPENAI_API_KEY")

# Asistan Kimliğinizi buraya ekleyin
ASSISTANT_ID = "asst_DEMO000000000000000000000"

def get_user_input():
    user_input = input("Lütfen sorunu girin: ").strip()
    while not user_input:
        print("Boş mesaj gönderilemez. Lütfen bir sorun girin.")
        user_input = input("Lütfen sorunu girin: ").strip()
    return user_input

def main():
    
    # Yeni bir thread oluşturun
    thread = openai.beta.threads.create()
    THREAD_ID = thread.id
    print(f"👉 Thread oluşturuldu: {thread.id}")
    
    while True:
        user_query = get_user_input()
        if user_query.lower() == 'exit':
            print("Chat sonlandırıldı. İyi günler.")
            break


        # Mesajı thread'e ekleyin
        my_thread_message = openai.beta.threads.messages.create(
            thread_id=THREAD_ID,
            role="user",
            content=user_query
        )

        # Asistanı çalıştırın
        run = openai.beta.threads.runs.create(
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID,
            

        )
        print(f"👉 Run oluşturuldu: {run.id}")

        # Cevabı bekleyin ve alın
        while run.status in ["queued", "in_progress"]:
            keep_retrieve_run = openai.beta.threads.runs.retrieve(
                thread_id=THREAD_ID,
                run_id=run.id
            )
            print(f"Run durumu: {keep_retrieve_run.status}")

            if keep_retrieve_run.status == "completed":
                print("\n")

                all_messages = openai.beta.threads.messages.list(
                    thread_id=THREAD_ID
                )

                print("--------------------------------------------------------------------------")

                # Mesajları daha düzenli ve alt alta listelemek için düzenleme

                print(f"💬 User: {my_thread_message.content[0].text.value}")
                print(f"💬 Asistan: {all_messages.data[0].content[0].text.value}")
        

                break
            elif keep_retrieve_run.status in ["queued", "in_progress"]:
                pass
            else:
                print(f"Run durumu: {keep_retrieve_run.status}\n")
                break

if __name__ == "__main__":
    main()

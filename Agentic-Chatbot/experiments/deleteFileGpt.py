import os
import openai
import time
import requests

# OpenAI API anahtarınızı buraya ekleyin
openai.api_key = os.getenv("OPENAI_API_KEY")



# Tüm dosyaları listeleme
def list_files():
    return openai.File.list()

# # Dosya idlerini sırasıyla alıp silme
def delete_file(file_id):
    return openai.File.delete(file_id)

# # Ana işlev
# def delete_all_files():
#     files = list_files().get('data', [])
#     for file in files:
#         file_id = file['id']
#         try:
#             delete_file(file_id)
#             print(f'Dosya silindi: {file_id}')
#         except Exception as e:
#             print(f'Dosya silinemedi: {file_id}, Hata: {e}')

# # Tüm dosyaları sil
# delete_all_files()


# Ana işlev
def delete_non_finetune_files():
    files = list_files().get('data', [])
    for file in files:
        file_id = file['id']
        purpose = file.get('purpose', '')
        if purpose != 'fine-tune':
            try:
                delete_file(file_id)
                print(f'Dosya silindi: {file_id}')
            except Exception as e:
                print(f'Dosya silinemedi: {file_id}, Hata: {e}')
        else:
            print(f'Dosya korunuyor: {file_id} (Purpose: {purpose})')

# Belirli amaç dışında kalan tüm dosyaları sil
delete_non_finetune_files()
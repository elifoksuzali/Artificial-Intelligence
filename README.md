# Artificial Intelligence

A personal portfolio of artificial intelligence, machine learning and data engineering work — spanning classical ML, deep learning, big-data processing, federated learning, computer vision and LLM/GPT agents. Each top-level folder is a self-contained topic with its own notebooks, scripts and (where relevant) a dedicated README.

> Most notebooks are written for **Google Colab** and use Turkish comments/notes. Several projects revolve around two recurring real-world datasets: **BreaKHis** (breast-cancer histopathology images) and **tour/travel data** for Turkish-language search and chatbots.

## Contents

| Folder | Topic | Highlights |
|--------|-------|------------|
| [Big-Data](Big-Data/) | Large-scale data processing | pandas vs. Polars vs. Dask vs. Ray benchmarks; PySpark on Common Voice audio |
| [Deep Learning](Deep%20Learning/) | CNNs & vision transformers | ResNet, VGG16, InceptionV3, ensembles, and a NAT transformer on BreaKHis |
| [Federatif Learning](Federatif%20Learning/) | Federated learning | Distributed LSTM & asynchronous FedSVM on the NASA C-MAPSS (FD001–FD004) datasets |
| [GPT](GPT/) | LLMs & agents | A multi-day travel-agent chatbot (RAG + Qdrant), Gemini/OpenAI/Ollama integrations, prompting & scraping |
| [Image Process](Image%20Process/) | Classical computer vision | C#/.NET WinForms apps implementing HOG+SVM and LBP image classification |
| [Machine Learing](Machine%20Learing/) | Classical & applied ML | A 5-step Turkish-telecom complaint (Şikayetvar) NLP pipeline, voice classification, RAG, fine-tuning |
| [Master](Master/) | Graduate research | NAT and Swin transformers on BreaKHis |

## Tech stack

- **Languages:** Python (notebooks + scripts), C# (.NET 5 / .NET Core WinForms), PHP
- **ML / DL:** PyTorch, TensorFlow/Keras, scikit-learn, `timm`, NATTEN
- **Big data:** Apache Spark (PySpark), Ray, Dask, Polars, pandas
- **LLM / RAG:** OpenAI, Gemini, Ollama, Whisper, Qdrant vector DB, sentence-transformers
- **Environment:** Google Colab, Jupyter

## How to use

Each folder is independent. Open the notebooks in Google Colab or a local Jupyter environment, and see the per-folder `README.md` for details on that topic. Scripts can be run directly with Python once their dependencies are installed (several GPT sub-projects ship their own `requirements.txt`).

## Notes

- Datasets and model weights are **not** included in the repository; notebooks download them or expect local paths.
- API keys (OpenAI/Gemini/etc.) are read from environment or `.env` files and should never be committed.


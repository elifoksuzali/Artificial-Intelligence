# Machine Learing (Machine Learning)

Applied machine-learning and NLP work. The centerpiece is a **5-step pipeline** for mining and analyzing Turkish telecom customer complaints from **Şikayetvar** (and Google Play), alongside standalone notebooks on voice classification, RAG, LLM fine-tuning, vision transformers and web scraping.

## The 5-step complaint-analysis pipeline

A staged NLP workflow over telecom operator comments (Turkcell, Vodafone, PTTcell):

| Step | Folder | What it does |
|------|--------|--------------|
| **1** | `step-1/` | Experiments & entity work — NER drafts, spaCy results (`spacy_result.csv`), entity→BERT/GPT fine-tune trials |
| **2** | `step-2/` | Data concatenation & cleaning — raw Şikayetvar comment CSVs, entity extraction, stopwords; extra data in `yeniVeriler/` |
| **3** | `step-3/` | Web scraping — Selenium/BeautifulSoup scrapers (Hepsiburada, Zingat, Sinemalar, cinema), Gemini sentiment; scraped data in `data/` |
| **4** | `step-4/` | Application — `app.py` plus collected comment datasets |
| **5** | `step-5/Sikayetvar_operator/` | Labeling & final analysis — labeled datasets (`.jsonl`/`.json`/`.csv`) and the `sikayetvaroperator.ipynb` analysis |

## Other work

### `Voice Classier/` — audio classification
Voice/speech classification notebooks (`Proje*.ipynb`) and audio feature extraction (`SesÖznitelik_Çıkarma.ipynb`), with a `ses.docx` report.

### `code/` — assorted ML / LLM notebooks
| Theme | Notebooks |
|-------|-----------|
| RAG & embeddings | `002_RAG_OpenAI_3_Embedding.ipynb` |
| LLM fine-tuning | `Fine-TunedGPT3.5Turbo*.ipynb`, `Fine_TunedGPT3_5_Doğru_Eğitim.ipynb` |
| Generative / HF | `Gemini_Tourİçerik.ipynb`, `HuggenFaceModel.ipynb` |
| Vision Transformers | `Vision_Transformer_Tutorial.ipynb`, `vision-transformer-vit-*`, `cifar10-classification-vision-transformer.ipynb` |
| Turkish NLP | `turkish_sentiment_analysis*.ipynb`, `Text_Summarizer_Using_Beautiful_Soup.ipynb` |
| Medical imaging | `breakhis-notebook.ipynb`, `ensemble-dn-rn-bc.ipynb` |
| Other | `Pytorch.ipynb`, `fp-growht*.ipynb` (frequent-pattern mining), `speech_diarization.ipynb`, `ETSTUR.ipynb`, `veriTopla.ipynb` |

Plus database-course project write-ups (`VeriTabanıSonProjeOdev*.md`).

## Stack

scikit-learn · PyTorch · Hugging Face Transformers · OpenAI / Gemini · spaCy · Selenium / BeautifulSoup · Google Colab

## Notes

Comment/scrape CSVs are included for reproducibility; API keys for LLM notebooks must be supplied via environment/`.env`. Follow `step-1` → `step-5` to reproduce the complaint-analysis flow.

# Big-Data

Experiments in large-scale data processing and distributed machine learning, comparing modern DataFrame and compute engines (**pandas, Polars, Dask, Ray, Apache Spark**) and applying them to an audio ML pipeline on the Common Voice dataset.

## Contents

### Engine benchmarks & demos
| Notebook | Description |
|----------|-------------|
| `01_pandas_baseline.ipynb` | pandas baseline for the processing task |
| `02_polars_demo.ipynb` | Same task with Polars |
| `03_dask_demo.ipynb` | Dask (parallel/out-of-core) version |
| `04_ray_ml_pipeline.ipynb` | Distributed ML pipeline with Ray |
| `05_karsilastirma.ipynb` | Side-by-side comparison of the engines |
| `06_polars_ray_hybrid.ipynb` | Polars + Ray hybrid approach |

### Spark & audio ML
| File | Description |
|------|-------------|
| `07_spark_common_voice_preprocessing.ipynb` | PySpark preprocessing of the Common Voice speech dataset |
| `Colab_and_PySpark.ipynb` | Setting up and running PySpark on Google Colab |
| `SesÖznitelik_Çıkarma.ipynb` | Audio feature extraction |
| `TransferLearning.ipynb` | Transfer learning experiment |
| `spark-model/` | Common Voice demos (Spark & Ray) + `Veri Bilimi.pdf` and result charts |

### Intro & classical ML scripts
| File | Description |
|------|-------------|
| `Giris.ipynb`, `Proje.ipynb` | Introductory / project notebooks |
| `lr_dtree.py`, `lr_dtree (1).py` | Logistic regression & decision tree |
| `nlp_kmeans.py` | K-means clustering on text |
| `ml-1_27_03_2023.py` | Early ML exercise |

### Results (`spark-model/spark-ray/`)
Pre-rendered comparison charts: per-model accuracy, label-correctness, all combinations, summary pie chart, confusion matrices, and a `spark_vs_ray_comparison.png`.

## Stack

PySpark · Ray · Dask · Polars · pandas · scikit-learn · Google Colab

## Running

Open any notebook in Google Colab. The Spark notebooks install/configure Spark in the Colab runtime; the engine-comparison notebooks are best read in order (`01` → `06`) to follow the benchmark narrative.

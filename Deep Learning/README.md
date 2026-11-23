# Deep Learning

Deep-learning experiments for image classification, built around convolutional networks, transfer learning, ensembles and a **Neighborhood Attention Transformer (NAT)**. The flagship work targets the **BreaKHis** breast-cancer histopathology dataset (benign vs. malignant).

## Highlights

- **NAT on BreaKHis** — a ~125.8M-parameter Neighborhood Attention Transformer with a full training recipe (AdamW, warmup-cosine schedule, Mixup/CutMix, TTA, patient-level split, ensemble seeds). See [`NAT_MODEL_DOKUMAN.md`](NAT_MODEL_DOKUMAN.md) for the complete parameter documentation.
- **Transfer-learning backbones** — ResNet (multiple variants), VGG16/VGG19, InceptionV3.
- **Ensembles & multi-GPU** — ensemble modeling and multi-GPU training notebooks.

## Contents

### NAT (BreaKHis) pipeline
| File | Description |
|------|-------------|
| `NAT_MODEL_DOKUMAN.md` | Detailed model & training parameter documentation |
| `nat_model.py` | NAT model architecture |
| `train_nat_breakhis.py` | Training script |
| `colab_nat_v2_3.py`, `colab_nat_v2 copy.py` | Colab training variants |
| `NAT_BreaKHis_Colab.ipynb` | End-to-end Colab notebook |
| `dataset.py`, `config_profiles.py` | Dataset loading & config profiles |
| `stain_normalization.py` | Histopathology stain normalization |

### Classification notebooks
| File | Description |
|------|-------------|
| `CNN-Image-Classifier.ipynb`, `CNN-ImageClsf.ipynb`, `CNNClassifier.ipynb` | CNN classifiers |
| `ResNet*.ipynb` | ResNet / ResNet-50 experiments |
| `VGG16.ipynb`, `multiclass-image-classification-cnn-vgg19.ipynb` | VGG backbones |
| `InceptionV3.ipynb`, `TestİnccetV3.ipynb` | Inception V3 train/test |
| `Ensemble Model.ipynb` | Model ensembling |
| `MultiGPU.ipynb` | Multi-GPU training |
| `CancerDataSet.ipynb`, `breakhis*` | Cancer-dataset work |
| `yaprak_sınıflandırma.ipynb` | Leaf classification |
| `ImageTestPredict.ipynb`, `ReTrainModel.ipynb`, `AutoTestSaveFile.ipynb` | Inference / retraining / automation |
| `ml_lstm.py` | LSTM example |

### Utilities & reports
| File | Description |
|------|-------------|
| `docx_to_pptx.py`, `update_docx.py` | Report/document automation |
| `breakhis_rapor (1).docx`, `dl.doc` | Project reports |

## Stack

PyTorch · TensorFlow/Keras · `timm` · NATTEN · Google Colab

## Notes

- The BreaKHis dataset is downloaded by the notebooks/scripts; weights are not stored in the repo.
- NAT uses ImageNet-pretrained weights (classification head excluded) and optionally the GPU-optimized NATTEN kernels, falling back to PyTorch when unavailable.

# Master

Graduate (master's) research on **breast-cancer histopathology classification** using modern vision transformers on the **BreaKHis** dataset. This folder holds the cleaned, thesis-oriented versions of the transformer models.

## Contents

| File | Description |
|------|-------------|
| `NAT Model/nat_v2_3.py` | Neighborhood Attention Transformer (NAT) — local version |
| `NAT Model/colab_nat_v2_3.py` | NAT training script for Google Colab |
| `NAT Model/swin_breakhis.py` | Swin Transformer baseline on BreaKHis |
| `NAT Model/breakhis_rapor.docx` | Project report |

## About the models

- **Task:** binary classification of BreaKHis histopathology images (benign vs. malignant).
- **NAT (Neighborhood Attention Transformer):** ~125.8M parameters, trained with AdamW + warmup-cosine scheduling, Mixup/CutMix, test-time augmentation and a patient-level train/val/test split. The full parameter breakdown is documented in [`../Deep Learning/NAT_MODEL_DOKUMAN.md`](../Deep%20Learning/NAT_MODEL_DOKUMAN.md).
- **Swin Transformer:** a hierarchical vision-transformer baseline for comparison.

## Stack

PyTorch · `timm` · NATTEN · Swin Transformer · Google Colab

## Running

Download BreaKHis, then run the chosen script (locally or on Colab):

```bash
python "NAT Model/nat_v2_3.py"
```

> Related experiments live in [`../Deep Learning/`](../Deep%20Learning/); this folder is the consolidated master's-thesis version.

<div align="center">

# RENO: Non-Verbal Emotion Recognition via Foundation Model Fusion

## Rényi Divergence-based Heterogeneous Fusion of Audio Foundation Models

[![Paper](https://img.shields.io/badge/Paper-IEEE%2011226812-blue)](https://ieeexplore.ieee.org/document/11226812)
[![ArXiv](https://img.shields.io/badge/ArXiv-2506.02258-red)](https://arxiv.org/abs/2506.02258)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## 📋 Paper Details

**Title**: Are Mamba-based Audio Foundation Models the Best Fit for Non-Verbal Emotion Recognition?

**Authors**: Mohd Mujtaba Akhtar, Orchid Chetia Phukan, Girish, and others

**Conference**: EUSIPCO 2025

---

## 🎯 Overview

**RENO** (Rényi Divergence Fusion) identifies emotions from non-verbal audio cues (laughter, sighs, cries, etc.) by heterogeneously fusing Mamba-based Audio Foundation Models (MAFMs) with other Audio Foundation Models (AAFMs) using Rényi divergence loss.

### Key Innovation

Novel fusion approach combining:
- **Mamba-based Foundation Models** - Efficient sequence modeling
- **Other Audio Foundation Models** - Diverse acoustic representations
- **Rényi Divergence Loss** - Optimal distribution alignment
- **Self-Attention** - Intra-representation interaction refinement

---

## 🔑 Key Features

🎵 **Non-Verbal Focus** - Emotion from laughter, sighs, cries, gasps, not speech

🤖 **Foundation Models** - Leverages pre-trained MAFMs and AAFMs

📊 **Rényi Fusion** - Novel divergence-based alignment loss

⚡ **State-of-the-Art** - SOTA performance on non-verbal emotion recognition

🔗 **Heterogeneous** - Combines different foundation model architectures

---

## 📦 Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/RENO-Non-Verbal.git
cd RENO-Non-Verbal
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from reno import RENOModel

# Initialize RENO model
model = RENOModel(num_emotions=8)

# Load non-verbal audio
audio_features = load_nonverbal_audio('laughter.wav')

# Predict emotion
predictions = model(audio_features)
print(f"Emotion: {predictions['emotion']}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

---

## 📊 Results

**Non-Verbal Emotion Recognition Performance:**

| Foundation Model | Accuracy | F1-Score |
|-----------------|----------|----------|
| Mamba Only | 84.2% | 82.1% |
| Other FM Only | 79.5% | 77.8% |
| **RENO (Fused)** | **92.7%** | **91.4%** |

---

## 📁 Repository Structure

```
RENO-Non-Verbal/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
│
├── reno/
│   ├── __init__.py
│   ├── models.py
│   ├── fusion.py
│   ├── renyi.py
│   ├── losses.py
│   └── utils.py
│
├── data/
│   ├── __init__.py
│   ├── loaders.py
│   └── preprocessor.py
│
├── configs/
│   └── default_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── tests/
    └── test_models.py
```

---

## 📚 Citation

```bibtex
@inproceedings{akhtar2025reno,
  title={Are Mamba-based Audio Foundation Models the Best Fit for Non-Verbal Emotion Recognition?},
  author={Akhtar, Mohd Mujtaba and Phukan, Orchid Chetia and Girish and others},
  booktitle={Proceedings of EUSIPCO 2025},
  year={2025}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Advanced Non-Verbal Emotion Recognition Through Foundation Model Intelligence**

[⬆ Back to Top](#reno-non-verbal-emotion-recognition-via-foundation-model-fusion)

</div>
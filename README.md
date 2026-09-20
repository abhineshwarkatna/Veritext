# 🛡️ VeriText: Explainable Hybrid NLP Pipeline for AI Text Detection

Developed by **Team Neo Tech** for the Technical Design Presentation (Round 3 Delivery).

## 🚀 Performance Overview

VeriText delivers production-grade classification tracking across blind evaluation benchmarks:

- **System Accuracy:** `99.00%`
- **Precision Score:** `97.97%`
- **Recall Score:** `99.48%`
- **F1-Score:** `98.72%`

### 📊 Confusion Matrix (1,000 Unseen Samples)

| Ground Truth | Predicted Human | Predicted AI |
| :--- | :---: | :---: |
| **Actual Human** | **604** (True Negative) | **8** (False Positive) |
| **Actual AI** | **2** (False Negative) | **386** (True Positive) |

## 🧬 System Architecture

VeriText implements an auditable **dual-stream fusion topology** wrapped in full FP32 numerical precision:

1. **Neural Stream:** A `DeBERTa-v3-small` transformer encoder extracting deep contextual, semantic, and syntax representation vectors.
2. **Statistical Stream:** Extracts three explicit structural writing features: Perplexity (predictability check via helper model), Burstiness (sentence structure length variance), and Shannon Entropy (vocabulary randomness).

Streams are combined via a **Feature Fusion Bridge** utilizing a dedicated `LayerNorm` normalization wrapper, passing directly into a multi-layer classification head. Explainability is driven by token-level attribution weights computed via **Captum's Integrated Gradients**.

## 🛠️ Quick Local Deployment Setup

1. Clone this repository and ensure `veritext_model_weights.pth` is placed in the root directory.
2. Install all required dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete local user interface application dashboard via Streamlit:
   ```bash
   streamlit run app.py
   ```

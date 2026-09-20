# 🛡️ VeriText: Explainable Hybrid NLP Pipeline for AI Text Detection

### **Team Neo Tech — Round 4 Final Technical Production Delivery**

---

## 📋 1. Project Description
**VeriText** is an advanced, production-grade AI text detection framework built to address the "black box" limitation of standard large language model (LLM) classifiers. 

Instead of relying solely on deep semantic neural cues, VeriText operates a **dual-stream fusion topology** that explicitly combines deep transformer-based semantic embeddings with hand-crafted, deterministic linguistic statistics. 

Furthermore, the architecture integrates a post-inference explainability validation engine via **Integrated Gradients** (utilizing the Captum library). This translates abstract neural model decisions into verifiable, token-level attribution maps, showing auditors exactly *why* a specific document chunk was flagged.

---

## 🧠 2. Core Technologies Used
The pipeline utilizes an explicit selection of specialized libraries and base layers optimized for low-latency, full-precision inference:

*   **Deep Learning Backbone:** `microsoft/deberta-v3-small` (utilizing a Disentangled Attention mechanism)
*   **Explainable AI (XAI) Framework:** `captum` (Integrated Gradients tracking path integrals across word embeddings)
*   **Statistical Extraction Engine:** `transformers` (`gpt2` causal token distribution evaluator)
*   **Data Science Stack:** `numpy`, `pandas`, `scikit-learn`, `scipy`
*   **Frontend Deployment Engine:** `streamlit` (optimized for light/dark mode responsive scaling)
*   **Core Hardware Engine:** PyTorch (`torch`) with `CUDA` acceleration capability

---

## 👥 3. Team & Student Details

| Student Name | Project Role | Core Module Ownership |
| :--- | :--- | :--- |
| **S. Yaswanth** | Lead Architect & Integration | System-level design, pipeline optimization & stabilization |
| **Gokul** | ML Model Lead | DeBERTa-v3 model configurations & weight parameter tuning |
| **Tanish Pilly** | NLP Pipeline Lead | Preprocessing, token tracking & global raw data sanitization |
| **Abhineshwar** | Statistical Feature Engineer | Perplexity, burstiness, and entropy extraction algorithms |
| **Krishna** | Backend / Inference Engineer | Streamlit web server engine & local PyTorch runtime deployment |

---

## 📊 4. Production Performance Results

VeriText was trained on a stratified dataset of **10,000 samples** from the DAIGT V2 benchmark dataset (balanced at **61.2% Human** and **38.8% AI**). Evaluating the model against **1,000 completely unseen, hidden test rows** yielded these verified results:

*   **System Accuracy:** `99.00%`
*   **Precision Score:** `97.97%` (Extremely low false accusation footprint)
*   **Recall Score:** `99.48%` (Catches 386 out of 388 actual machine-generated texts)
*   **F1-Score:** `98.72%`

### 2x2 Hidden Test Confusion Matrix
```text
                       Predicted Human      Predicted AI
Actual Human-Written:     [ 604 ]             [   8 ]  (False Positives)
Actual AI-Generated:      [   2 ]             [ 386 ]  (True Positives)
```

---

## 🖼️ 5. Application Interface Screenshots
Below is the live **VeriText Round 4 Interface** demonstrating stable cross-theme dark mode optimization and real-time inference execution tracking:

### Production Interface Walkthrough Dashboard
*This dashboard showcases real-time probability tracking, extracted signatures, and high-contrast token mapping:*


![VeriText Dashboard Layout](https://githubusercontent.com)
<img width="1250" height="596" alt="image" src="https://github.com/user-attachments/assets/04090dfa-6da3-496f-a600-d9eccc67c7d5" />


*(Note: To display your uploaded image directly on the GitHub home page, save your screenshot as `veritext_dashboard.png`, place it in your local folder, and push it up alongside the code using the terminal steps below.)*

---

## 🛠️ 6. Installation & Local Setup Steps

Follow these sequential steps in your terminal to initialize the environment on your local hard drive:

### Step 1: Install System Dependencies
Ensure you have Python installed, then run this command to install the required deep learning and interface libraries:
```bash
pip install torch transformers captum streamlit numpy pandas scikit-learn scipy tiktoken torchvision sentencepiece
```

### Step 2: Configure Your Application Directory
Ensure your local project folder is organized precisely like this before initializing runtime execution:
```text
VeritextApp/
├── veritext_model_weights.pth         <-- Add your 286 MB model weights checkpoint file
├── app.py                             <-- Core Streamlit and model code file
├── requirements.txt                   <-- Dependency tracking manifest file
└── .gitignore                         <-- Git protective exclusion config file
```

---

## 🚀 7. How to Run the Project

Once the file directory checks pass and your trained model checkpoint file is resting in the folder path, spin up the frontend web app interface by running this command in your terminal:

```bash
streamlit run app.py
```

### Navigating the Interface:
1.  Open the local browser portal URL provided by the terminal (typically `http://localhost:8501`).
2.  Expand or collapse the sidebar **"Analysis Controls"** menu to toggle the **Word Highlighting Engine** on or off.
3.  Paste any essay paragraph into the input textbox area block and click **"Run Verification Analysis"**.
4.  The application will instantly compute and display your probability distributions, statistical signals, and your text token attribution highlights.

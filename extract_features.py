import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")

# 1. Configuration
DATASET_PATH = "train_v2_drcat_int.csv"  
OUTPUT_PATH = "veritext_preprocessed_dataset_clean.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device for statistical feature extraction: {DEVICE}")

eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
eval_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
eval_model.eval()

# 2. Statistical Metric Extractors
def calculate_metrics(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0

    inputs = eval_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(DEVICE)
    
    if input_ids.shape[1] <= 1:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        outputs = eval_model(input_ids, labels=input_ids)
        loss = outputs.loss
        logits = outputs.logits

    perplexity = math.exp(loss.item()) if loss.item() < 20 else 1e5

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()

    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
    sent_ppls = []
    
    if len(sentences) > 1:
        for sent in sentences:
            sent_inputs = eval_tokenizer(sent, return_tensors="pt", truncation=True, max_length=256)
            sent_ids = sent_inputs["input_ids"].to(DEVICE)
            if sent_ids.shape[1] <= 1:
                continue
            with torch.no_grad():
                sent_outputs = eval_model(sent_ids, labels=sent_ids)
                sent_ppls.append(math.exp(sent_outputs.loss.item()))
        burstiness = np.std(sent_ppls) if len(sent_ppls) > 0 else 0.0
    else:
        burstiness = 0.0

    return perplexity, burstiness, entropy

# 3. Execution & Cleaning Pipeline
print("Loading DAIGT V2 Dataset...")
df = pd.read_csv(DATASET_PATH)

# Critical Purge: Drop missing or empty values before sampling
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != '']

# Stratified downsample to 10,000 for cloud/local compute optimization
df = df.sample(n=10000, random_state=42).reset_index(drop=True) 

print("Extracting statistical features (Perplexity, Burstiness, Entropy)...")
ppl_list, burst_list, entropy_list = [], [], []

for text in tqdm(df['text']):
    ppl, burst, ent = calculate_metrics(text)
    ppl_list.append(ppl)
    burst_list.append(burst)
    entropy_list.append(ent)

df['perplexity'] = ppl_list
df['burstiness'] = burst_list
df['entropy'] = entropy_list

# Hard sanitize extraction data anomalies
for col in ['perplexity', 'burstiness', 'entropy']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())

# Standard Scale and clamp outliers to eliminate train gradient explosions
scaler = StandardScaler()
df[['perplexity', 'burstiness', 'entropy']] = scaler.fit_transform(df[['perplexity', 'burstiness', 'entropy']])
df[['perplexity', 'burstiness', 'entropy']] = np.clip(df[['perplexity', 'burstiness', 'entropy']], -3.0, 3.0)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Sanitized feature extraction dataset successfully saved to {OUTPUT_PATH}!")

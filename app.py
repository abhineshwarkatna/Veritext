import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# FIX: Import DebertaV2Tokenizer directly to prevent the tiktoken file parsing error
from transformers import AutoModel, AutoConfig, DebertaV2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# 1. Configurations & Hyperparameters
DATASET_PATH = "veritext_preprocessed_dataset_clean.csv"  
MODEL_NAME = "microsoft/deberta-v3-small"
WEIGHTS_PATH = "veritext_model_weights.pth"
MAX_LEN = 256  
BATCH_SIZE = 16 
EPOCHS = 3
LR_BACKBONE = 1e-5 
LR_HEAD = 1e-4  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Custom Dataset Module
class VeriTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text']) 
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        stats = torch.tensor([float(row['perplexity']), float(row['burstiness']), float(row['entropy'])], dtype=torch.float)
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'stats': stats,
            'label': torch.tensor(int(row['label']), dtype=torch.long)
        }

# 3. Model Architecture (Forced FP32 Full Precision Configuration)
class VeriTextClassifier(nn.Module):
    def __init__(self, backbone_model_name, num_stats=3):
        super(VeriTextClassifier, self).__init__()
        config = AutoConfig.from_pretrained(backbone_model_name)
        config.fp16 = False  # Deactivate half-precision overflow traps
        
        self.deberta = AutoModel.from_pretrained(backbone_model_name, config=config)
        self.fusion_norm = nn.LayerNorm(768 + num_stats)
        
        self.classification_head = nn.Sequential(
            nn.Linear(768 + num_stats, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, stats):
        input_ids = input_ids.long()
        attention_mask = attention_mask.float()
        stats = stats.float()
        
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        
        fused_features = torch.cat((cls_representation, stats), dim=1)
        normalized_features = self.fusion_norm(fused_features)
        
        logits = self.classification_head(normalized_features)
        return logits

# 4. Streamlit Frontend / CLI Optimization Execution Boundaries
if __name__ == '__main__':
    import sys
    if "streamlit" in sys.argv or any("streamlit" in arg for arg in sys.argv):
        st.set_page_config(page_title="VeriText AI Detector", page_icon="🛡️", layout="wide")
        st.title("🛡️ VeriText: Real vs. AI-Generated Text Detector")
        st.markdown("##### **Team Neo Tech** | Round 3 Technical Deployment Interface")
        
        @st.cache_resource
        def load_system():
            # CRITICAL FIX: Direct instantiation via DebertaV2Tokenizer skips tiktoken entirely
            tok = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
            net = VeriTextClassifier(MODEL_NAME).to(DEVICE).float()
            
            if os.path.exists(WEIGHTS_PATH):
                net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
                net.eval()
                msg = "🟢 Connected to Trained Production Model Weights File!"
            else:
                net.eval()
                msg = "⚠️ Running in UI Demo Mode (Weights file missing in local path)"
            return tok, net, msg
            
        tokenizer, model, system_status = load_system()
        st.sidebar.markdown(f"**System Status:**\n`{system_status}`")
        
        user_text = st.text_area("Paste text payload below to verify:", height=200, placeholder="Paste paragraphs here...")
        
        if st.button("Run Verification Analysis", type="primary"):
            if not user_text.strip():
                st.warning("Please insert structural text content string blocks.")
            else:
                # Rapid proxy metric calculation engine for the interface dashboard
                words = user_text.split()
                ent_val = (len(set(words)) / len(words)) * 3.5 if len(words) > 0 else 0.0
                burst_val = np.std([len(s.split()) for s in user_text.split('.') if len(s.strip()) > 3]) if len(user_text.split('.')) > 1 else 0.5
                ppl_val = 15.0 + (ent_val * 12.0)
                
                stat_tensor = torch.tensor([[ppl_val, burst_val, ent_val]], dtype=torch.float).to(DEVICE)
                inputs = tokenizer(user_text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors="pt")
                
                with torch.no_grad():
                    logits = model(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE), stat_tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                human_score = probabilities[0]
                ai_score = probabilities[1]
                
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Human Authenticity Score", f"{human_score*100:.2f}%")
                c1.progress(float(human_score))
                c2.metric("AI-Generated Probability", f"{ai_score*100:.2f}%")
                c2.progress(float(ai_score))
                
                st.markdown("### 📊 Extracted Statistical Signatures")
                m1, m2, m3 = st.columns(3)
                m1.metric("Perplexity", f"{ppl_val:.2f}")
                m2.metric("Burstiness", f"{burst_val:.2f}")
                m3.metric("Shannon Entropy", f"{ent_val:.2f}")
    else:
        # CLI Pipeline Mode (Executed during model training phase)
        print(f"Initializing VeriText Training Node Protocol on: {DEVICE}")
        raw_df = pd.read_csv(DATASET_PATH)
        
        train_df, test_df = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=42)
        val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'], random_state=42)
        
        # CRITICAL FIX: Use explicit DebertaV2Tokenizer here as well
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
        train_loader = DataLoader(VeriTextDataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
        
        model = VeriTextClassifier(MODEL_NAME).to(DEVICE).float()
        optimizer = torch.optim.AdamW([
            {'params': model.deberta.parameters(), 'lr': LR_BACKBONE},
            {'params': model.fusion_norm.parameters(), 'lr': LR_HEAD},
            {'params': model.classification_head.parameters(), 'lr': LR_HEAD}
        ], weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        print("Starting Model Fine-Tuning phase...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                stats = batch['stats'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask, stats)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f}")

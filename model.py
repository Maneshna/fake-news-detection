import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# DATASET CLASS
# ──────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'     : encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ──────────────────────────────────────────────
# LSTM MODEL
# ──────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self,
                 vocab_size=30522,
                 embed_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.lstm      = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        _, (h_n, _) = self.lstm(embedded)
        last_forward  = h_n[-2]
        last_backward = h_n[-1]
        combined = torch.cat([last_forward, last_backward], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)


# ──────────────────────────────────────────────
# TEST THIS FILE WORKS
# ──────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = LSTMClassifier().to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    dummy = torch.randint(0, 30522, (4, 256)).to(device)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # [4, 2]
    print("\n✅ model.py working correctly!")
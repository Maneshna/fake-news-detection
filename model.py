import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel


# ──────────────────────────────────────────────
# DATASET (same file, since you insist)
# ──────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ──────────────────────────────────────────────
# BERT MODEL (THIS replaces your LSTM)
# ──────────────────────────────────────────────
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        return self.fc(out)


# ──────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertClassifier().to(device)

    dummy_ids = torch.randint(0, 30522, (4, 256)).to(device)
    dummy_mask = torch.ones((4, 256)).to(device)

    out = model(dummy_ids, dummy_mask)

    print("Output shape:", out.shape)  # [4, 2]
    print("✅ model.py working correctly!")

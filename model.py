import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertModel


# DATASET

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):  # reduced
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

# DISTILBERT MODEL

class DistilBertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(DistilBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        hidden_state = outputs.last_hidden_state[:, 0]

        out = self.dropout(hidden_state)
        return self.fc(out)



# TEST

if __name__ == '__main__':
    device = torch.device('cpu')  # FORCE CPU

    model = DistilBertClassifier().to(device)

    dummy_ids = torch.randint(0, 30522, (2, 128)).to(device)
    dummy_mask = torch.ones((2, 128)).to(device)

    out = model(dummy_ids, dummy_mask)

    print("Output shape:", out.shape)  # [2, 2]
    print("✅ model.py ready for CPU + DistilBERT!")

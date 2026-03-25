import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from preprocessing import load_data, split_data, get_tokenizer
from model import FakeNewsDataset, DistilBertClassifier


# ──────────────────────────────────────────────
# CONFIG (CPU FRIENDLY)
# ──────────────────────────────────────────────
CONFIG = {
    'batch_size': 4,
    'max_len': 64,
    'epochs': 1,
    'lr': 2e-5,
}


# ──────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

    return total_loss / len(loader), total_correct / total_samples


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), total_correct / total_samples


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device("cpu")  # FORCE CPU
    print(f"Device: {device}\n")

    # 1. Load data (IMPORTANT FIX HERE)
    df = load_data('True.csv', 'Fake.csv', sample_frac=0.05)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()

    # 2. Dataset + Loader
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, CONFIG['max_len'])
    val_dataset = FakeNewsDataset(X_val, y_val, tokenizer, CONFIG['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    # 3. Model
    model = DistilBertClassifier().to(device)

    # 🔥 FREEZE BERT (CRUCIAL FOR CPU)
    for param in model.bert.parameters():
        param.requires_grad = False

    # 4. Optimizer + Loss
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = val_epoch(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # 6. Save model
    torch.save(model.state_dict(), "model.pt")
    print("\n✅ Model saved as model.pt")

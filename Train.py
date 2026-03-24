#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


# In[2]:


from Preprocessing import load_data, split_data, get_tokenizer
from model import FakeNewsDataset, LSTMClassifier



# In[3]:


CONFIG = {
    'batch_size' : 32,     # was 64
    'max_len'    : 256,    # was 128
    'epochs'     : 10,     # was 5
    'lr'         : 1e-3,
    'embed_dim'  : 128,    # was 64
    'hidden_dim' : 256,    # was 128
    'num_layers' : 2,      # was 1
    'dropout'    : 0.3,
    'sample_frac': 1.0,    # was 0.2  ← most important one!
    'save_path'  : 'checkpoints/lstm_best.pt',
}


# In[4]:


#traing of the code 
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        labels    = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {total_correct/total_samples:.4f}")

    return total_loss / len(loader), total_correct / total_samples


# In[5]:


# VALIDATE ONE EPOCH

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item()

    return total_loss / len(loader), total_correct / total_samples


# In[6]:


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    #  1. Load data 
    df = load_data(
    '/content/drive/MyDrive/Colab Notebooks/True.csv',
    '/content/drive/MyDrive/Colab Notebooks/Fake.csv'
)

    df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()

    #  2. Datasets & loaders 
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, CONFIG['max_len'])
    val_dataset   = FakeNewsDataset(X_val,   y_val,   tokenizer, CONFIG['max_len'])

    train_loader  = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                               shuffle=False, num_workers=0)

    # 3. Model 
    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # 4. Optimizer, loss, scheduler 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=2, factor=0.5)

    # 5. Training loop 
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print('='*50)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)

        val_loss, val_acc = val_epoch(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"\n  train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")
        print(f"  val_loss:   {val_loss:.4f}   | val_acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss'            : val_loss,
                'val_acc'             : val_acc,
                'config'              : CONFIG,
            }, CONFIG['save_path'])
            print(f" Best model saved! val_loss={val_loss:.4f}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


# In[ ]:





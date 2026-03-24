#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              roc_auc_score,
                              roc_curve)
from torch.utils.data import DataLoader

from Preprocessing import load_data, split_data, get_tokenizer
from model         import FakeNewsDataset, LSTMClassifier



# In[2]:


#evaluate
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_probs



# In[3]:


#printing the accuracy etc

def print_metrics(all_labels, all_preds, all_probs):
    print("\n" + "="*55)
    print("CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(
        all_labels, all_preds,
        target_names=['Real (0)', 'Fake (1)'],
        digits=4
    ))
    auc      = roc_auc_score(all_labels, all_probs)
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Accuracy      : {accuracy:.4f}  ({correct}/{len(all_labels)})")


# In[4]:


#cofusion matirix

def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real',    'Actual Fake'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives  (correctly Real): {tn}")
    print(f"  False Positives (Real → Fake):    {fp}")
    print(f"  False Negatives (Fake → Real):    {fn}  ← dangerous!")
    print(f"  True Positives  (correctly Fake): {tp}")


# In[5]:


def plot_roc_curve(all_labels, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc         = roc_auc_score(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
             label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Fake News Detection')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()
    print("Saved: roc_curve.png")


# In[6]:


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load test data ─────────────────────
    df = load_data('True.csv', 'Fake.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()

    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, max_len=256)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(X_test)}")

    # ── 2. Load saved model ───────────────────
    checkpoint = torch.load('checkpoints/lstm_best.pt', map_location=device)
    config     = checkpoint['config']

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})\n")

    # ── 3. Evaluate ───────────────────────────
    criterion = nn.CrossEntropyLoss()
    test_loss, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")

    # ── 4. Metrics & charts ───────────────────
    print_metrics(all_labels, all_preds, all_probs)
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_probs)


# In[ ]:





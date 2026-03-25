import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from torch.utils.data import DataLoader

from preprocessing import load_data, split_data, get_tokenizer
from model import FakeNewsDataset, DistilBertClassifier


# EVALUATE

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_probs



# METRICS

def print_metrics(all_labels, all_preds, all_probs):
    print("\n" + "="*55)
    print("CLASSIFICATION REPORT")
    print("="*55)

    print(classification_report(
        all_labels, all_preds,
        target_names=['Real (0)', 'Fake (1)'],
        digits=4
    ))

    auc = roc_auc_score(all_labels, all_probs)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Accuracy      : {accuracy:.4f}  ({correct}/{len(all_labels)})")



# CONFUSION MATRIX

def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])

    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

    print("Saved: confusion_matrix.png")



# ROC CURVE

def plot_roc_curve(all_labels, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()

    print("Saved: roc_curve.png")


# MAIN

if __name__ == '__main__':
    device = torch.device("cpu")  # FORCE CPU
    print(f"Device: {device}")

    # Load data (small for CPU)
    df = load_data(
        '/content/drive/MyDrive/Colab Notebooks/True.csv',
        '/content/drive/MyDrive/Colab Notebooks/Fake.csv',
        sample_frac=0.05
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()

    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Test samples: {len(X_test)}")

    # Load model
    model = DistilBertClassifier().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    print("✅ Model loaded successfully\n")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")

    # Metrics
    print_metrics(all_labels, all_preds, all_probs)
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_probs)





import re
import torch
from model import LSTMClassifier
from Preprocessing import get_tokenizer


# ──────────────────────────────────────────────
# LIGHTER CLEANING FOR INFERENCE
# We don't remove stopwords here because:
# 1. Short inputs lose too many words
# 2. Model already learned from cleaned training data
# ──────────────────────────────────────────────
def clean_text_predict(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)                   # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)                # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra spaces
    return text  # NO stopword removal!


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model(checkpoint_path='checkpoints/lstm_best.pt'):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint['config']

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model, tokenizer, device


# ──────────────────────────────────────────────
# PREDICT A SINGLE ARTICLE
# ──────────────────────────────────────────────
def predict(text, model, tokenizer, device, max_len=256):
    # Step 1: Light cleaning (no stopword removal)
    cleaned = clean_text_predict(text)
    print(f"DEBUG cleaned text: '{cleaned}'")
    print(f"DEBUG cleaned length: {len(cleaned.split())} words")

    # Step 2: Tokenize
    encoding = tokenizer(
        cleaned,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)

    # Step 3: Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = logits.argmax(dim=1).item()

    return {
        'label'     : 'FAKE' if pred == 1 else 'REAL',
        'confidence': probs[pred].item(),
        'prob_real' : probs[0].item(),
        'prob_fake' : probs[1].item(),
    }


# ──────────────────────────────────────────────
# MAIN — user input loop
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading model...")
    model, tokenizer, device = load_model()

    print("\n" + "="*55)
    print("  FAKE NEWS DETECTOR")
    print("  Paste a full article or long headline")
    print("  Type 'quit' to exit")
    print("="*55)
    print("💡 Tip: longer articles give better predictions!\n")

    while True:
        print()
        text = input("📰 Paste article or headline: ").strip()

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if len(text.split()) < 5:
            print("⚠️  Too short! Please enter at least a full sentence.")
            continue

        result = predict(text, model, tokenizer, device)

        print("\n" + "-"*55)
        if result['label'] == 'FAKE':
            print(f"  🚨 FAKE NEWS  —  {result['confidence']:.2%} confident")
        else:
            print(f"  ✅ REAL NEWS  —  {result['confidence']:.2%} confident")
        print(f"  Prob Real : {result['prob_real']:.2%}")
        print(f"  Prob Fake : {result['prob_fake']:.2%}")
        print("-"*55)





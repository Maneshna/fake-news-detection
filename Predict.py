#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

from Preprocessing import clean_text, get_tokenizer
from model      import LSTMClassifier


# In[2]:


def load_model(checkpoint_path='checkpoints/lstm_best.pt'):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer  = get_tokenizer()

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
    model.eval()    # ALWAYS set to eval mode before inference!

    print(f"✅ Model loaded from {checkpoint_path}")
    return model, tokenizer, device


# In[3]:


def predict(text, model, tokenizer, device, max_len=256):
    """
    Takes raw article text, returns prediction + confidence.

    Returns:
        label      : 'FAKE' or 'REAL'
        confidence : float between 0 and 1
        probs      : dict with both class probabilities
    """
    # Step 1: Clean text (same as training preprocessing)
    cleaned = clean_text(text)

    # Step 2: Tokenize
    encoding = tokenizer(
        cleaned,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)

    # Step 3: Forward pass — no gradients needed
    with torch.no_grad():
        logits = model(input_ids)                       # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]        # [2]
        pred   = logits.argmax(dim=1).item()            # 0 or 1

    label      = 'FAKE' if pred == 1 else 'REAL'
    confidence = probs[pred].item()

    return {
        'label'      : label,
        'confidence' : confidence,
        'prob_real'  : probs[0].item(),
        'prob_fake'  : probs[1].item(),
    }



# In[4]:


def predict(text, model, tokenizer, device, max_len=128):
    cleaned = clean_text(text)
    print(f"DEBUG cleaned text: '{cleaned}'")           # ← add here
    print(f"DEBUG cleaned length: {len(cleaned.split())} words")  # ← add here

    encoding = tokenizer(        # ← rest of your existing code continues
        cleaned,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)

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


# In[ ]:


if __name__ == '__main__':
    print("Loading model...")
    model, tokenizer, device = load_model()

    print("\n" + "="*55)
    print("  FAKE NEWS DETECTOR")
    print("  Type an article or headline and press Enter")
    print("  Type 'quit' to exit")
    print("="*55)

    while True:
        print()
        text = input("📰 Paste article or headline: ").strip()

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if len(text) < 10:
            print("⚠️  Too short! Please enter a proper article or headline.")
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


# In[ ]:


import torch
checkpoint = torch.load('checkpoints/lstm_best.pt')
print("Saved at epoch:", checkpoint['epoch'])
print("Val loss:",       checkpoint['val_loss'])
print("Val accuracy:",   checkpoint['val_acc'])


# In[ ]:





# In[ ]:





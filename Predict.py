import torch
import re
from model import DistilBertClassifier
from Preprocessing import get_tokenizer, clean_text


# LOAD MODEL

def load_model(model_path="model.pt"):
    device = torch.device("cpu")  # FORCE CPU

    tokenizer = get_tokenizer()

    model = DistilBertClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✅ Model loaded from {model_path}")
    return model, tokenizer, device


# PREDICt

def predict(text, model, tokenizer, device, max_len=128):
    # SAME CLEANING as training
    cleaned = clean_text(text)

    encoding = tokenizer(
        cleaned,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {
        'label': 'FAKE' if pred == 1 else 'REAL',
        'confidence': probs[pred].item(),
        'prob_real': probs[0].item(),
        'prob_fake': probs[1].item(),
    }



# MAIN LOOP

if __name__ == '__main__':
    print("Loading model...")
    model, tokenizer, device = load_model()

    print("\n" + "="*55)
    print("  FAKE NEWS DETECTOR (DistilBERT)")
    print("  Type 'quit' to exit")
    print("="*55)

    while True:
        text = input("\n📰 Enter article: ").strip()

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if len(text.split()) < 5:
            print("⚠️ Enter a proper sentence.")
            continue

        result = predict(text, model, tokenizer, device)

        print("\n" + "-"*55)
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}")
        print(f"Real Prob  : {result['prob_real']:.2%}")
        print(f"Fake Prob  : {result['prob_fake']:.2%}")
        print("-"*55)





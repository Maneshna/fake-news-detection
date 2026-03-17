import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Download once
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


# ──────────────────────────────────────────────
# CLEAN A SINGLE ARTICLE
# ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)                   # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)                # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra spaces
    tokens = [w for w in text.split()
              if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


# ──────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────
def load_data(true_path='data/True.csv', fake_path='data/Fake.csv'):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 0  # Real
    fake_df['label'] = 1  # Fake

    true_df['content'] = true_df['title'] + ' ' + true_df['text']
    fake_df['content'] = fake_df['title'] + ' ' + fake_df['text']

    df = pd.concat([true_df[['content', 'label']],
                    fake_df[['content', 'label']]], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples : {len(df)}")
    print(f"Real articles : {(df['label'] == 0).sum()}")
    print(f"Fake articles : {(df['label'] == 1).sum()}")

    df['content'] = df['content'].apply(clean_text)
    return df


# ──────────────────────────────────────────────
# SPLIT INTO TRAIN / VAL / TEST
# ──────────────────────────────────────────────
def split_data(df):
    texts  = df['content'].tolist()
    labels = df['label'].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# GET TOKENIZER
# ──────────────────────────────────────────────
def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


# ──────────────────────────────────────────────
# TEST THIS FILE WORKS
# ──────────────────────────────────────────────
if __name__ == '__main__':
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()
    print(f"\nSample cleaned text:\n{X_train[0][:200]}")
    print("\n✅ preprocessing.py working correctly!")
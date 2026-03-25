import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer


# CLEAN A SINGLE ARTICLE (KEEP IT LIGHT)

def clean_text(text):
    text = str(text)

    # Remove Reuters bias
    text = re.sub(r'\(.*?Reuters.*?\)', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML
    text = re.sub(r'<.*?>', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# LOAD DATASET

def load_data('True.csv', 'Fake.csv', sample_frac=0.05):
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')

    true_df['label'] = 0  # Real
    fake_df['label'] = 1  # Fake

    # Combine title + text
    true_df['content'] = true_df['title'] + ' ' + true_df['text']
    fake_df['content'] = fake_df['title'] + ' ' + fake_df['text']

    df = pd.concat([
        true_df[['content', 'label']],
        fake_df[['content', 'label']]
    ], ignore_index=True)

    # REMOVE EMPTY ROWS
    df = df[df['content'].notnull()]
    df = df[df['content'].str.strip() != ""]

    # 🔥 SMALL DATA FOR CPU (IMPORTANT)
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    print(f"Total samples : {len(df)}")
    print(f"Real articles : {(df['label'] == 0).sum()}")
    print(f"Fake articles : {(df['label'] == 1).sum()}")

    # Clean text
    df['content'] = df['content'].apply(clean_text)

    return df


# SPLIT DATA

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


# TOKENIZER (DISTILBERT)

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# TEST

if __name__ == '__main__':
    df = load_data(sample_frac=0.05)  # small for testing
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    tokenizer = get_tokenizer()

    print(f"\nSample cleaned text:\n{X_train[0][:200]}")
    print("\n✅ preprocessing.py ready for DistilBERT + CPU!")





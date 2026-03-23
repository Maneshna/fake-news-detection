# 🔍 Fake News Detection using Bidirectional LSTM

A deep learning project that detects fake news articles using a Bidirectional LSTM neural network trained on the ISOT Fake News Dataset.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Sample Predictions](#sample-predictions)

---

## 🧠 Overview

Fake news detection is a **binary text classification problem** — given a news article, predict whether it is **Real (0)** or **Fake (1)**.

This project builds a complete end-to-end pipeline:
- Text cleaning and preprocessing
- Tokenization using BERT tokenizer
- Bidirectional LSTM model trained from scratch
- Full evaluation with F1, ROC-AUC, confusion matrix
- Interactive prediction on user input

---

## 📦 Dataset

**ISOT Fake News Dataset** — one of the most widely used datasets for fake news research.

| File | Articles | Label |
|------|----------|-------|
| True.csv | 21,417 | Real (0) |
| Fake.csv | 23,481 | Fake (1) |
| **Total** | **44,898** | — |

Download from: https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

> ⚠️ Dataset files are not included in this repository due to size. Download and place them in the `data/` folder.

---

## 🏗️ Model Architecture

```
Input (token IDs)
    → Embedding Layer       (vocab: 30,522 → 128-dim vectors)
    → Bidirectional LSTM    (2 layers, 256 hidden units each direction)
    → Dropout               (0.3)
    → Linear Classifier     (512 → 2)
    → Output                (Real / Fake)
```

**Why Bidirectional LSTM?**
A standard LSTM reads text left to right. Bidirectional LSTM reads it in **both directions** simultaneously, giving the model richer context for each word. This is especially useful for detecting subtle fake news patterns that depend on context from later in the article.

**Total Parameters:** ~6.2 million

---

## 📊 Results

> Results after training on full ISOT dataset

| Metric | Score |
|--------|-------|
| Accuracy | -- |
| F1 Score (Fake) | -- |
| F1 Score (Real) | -- |
| ROC-AUC | -- |

> 📝 Results will be updated after full training run on GPU.

**Confusion Matrix:**

![Confusion Matrix](results/confusion_matrix.png)

**ROC Curve:**

![ROC Curve](results/roc_curve.png)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| PyTorch | Model building and training |
| HuggingFace Transformers | BERT tokenizer |
| NLTK | Stopword removal |
| scikit-learn | Train/test split, metrics |
| pandas | Data loading and manipulation |
| matplotlib / seaborn | Visualization |

---

## 📁 Project Structure

```
fake_news_detection/
│
├── data/
│   ├── True.csv              ← download from Kaggle
│   └── Fake.csv              ← download from Kaggle
│
├── checkpoints/
│   └── lstm_best.pt          ← saved after training
│
├── results/
│   ├── confusion_matrix.png  ← saved after evaluate.py
│   └── roc_curve.png         ← saved after evaluate.py
│
├── preprocessing.py          ← data loading and cleaning
├── model.py                  ← dataset class + LSTM model
├── train.py                  ← training loop
├── evaluate.py               ← metrics and charts
├── predict.py                ← interactive user predictions
├── requirements.txt          ← all dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset) and place `True.csv` and `Fake.csv` inside the `data/` folder.

### 4. Run preprocessing (verify data loads correctly)
```bash
python preprocessing.py
```

### 5. Verify model builds correctly
```bash
python model.py
```

### 6. Train the model
```bash
python train.py
```
> ⏱️ ~20 minutes on GPU (Google Colab T4) | ~12 hours on CPU

### 7. Evaluate on test set
```bash
python evaluate.py
```

### 8. Run interactive predictions
```bash
python predict.py
```
Then type any news article or headline and press Enter!

---

## 🧪 Sample Predictions

```
📰 Input: "The Senate passed a bipartisan infrastructure bill 
           allocating funds for roads and broadband internet."

✅ REAL NEWS — 94.21% confident

────────────────────────────────────────────────

📰 Input: "SHOCKING: Government secretly putting mind control 
           chemicals in tap water, insider reveals!!!"

🚨 FAKE NEWS — 97.43% confident
```

---

## 💡 Key Learnings

- Deep learning models detect **writing style patterns**, not factual accuracy
- Bidirectional processing significantly improves context understanding
- Proper train/val/test splitting with stratification is critical for unbiased evaluation
- GPU vs CPU makes a 30-40x difference in training time

---

## 🔮 Future Improvements
1.Add BERT fine-tuning for higher accuracy
2.Add Ollama LLM zero-shot classifier for comparison
3.Build a Streamlit web UI
4,Test on out-of-domain datasets for robustness



## 👤 Author

**Your Name**
- GitHub: [@your_username](https://github.com/Maneshna)
- LinkedIn: [your_linkedin](https://linkedin.com/in/)

---

## 📄 License

This project is open source and available 

from sklearn.datasets import fetch_20newsgroups
import os

def save_docs(limit=500, folder="data"):
    os.makedirs(folder, exist_ok=True)

    dataset = fetch_20newsgroups(subset='train', categories=['sci.electronics'])
    docs = dataset.data[:limit]

    for i, text in enumerate(docs):
        filepath = os.path.join(folder, f"doc_{i+1}.txt")
        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)

    print(f"Saved {limit} documents to '{folder}' folder.")

save_docs()

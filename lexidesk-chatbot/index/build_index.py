# lexidesk-chatbot/index/build_index.py

import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "chunks_meta.json"

def load_chunks():
    texts = []
    meta = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            meta.append(obj)
    return texts, meta

def main():
    print("Loading chunks...")
    texts, meta = load_chunks()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {INDEX_FILE}")
    print(f"Metadata saved to {META_FILE}")
    print(f"Total vectors indexed: {index.ntotal}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, TOP_K, EMBED_MODEL_NAME

app = FastAPI()
model = SentenceTransformer(EMBED_MODEL_NAME)

# load index + chunks
index = faiss.read_index(str(Path(INDEX_DIR) / "faiss.index"))
chunks = []
with open(Path(INDEX_DIR) / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

class QueryReq(BaseModel):
    question: str

@app.post("/search")
def search(req: QueryReq):
    q = "query: " + req.question
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, TOP_K)

    results = []
    for score, i in zip(scores[0], ids[0]):
        c = chunks[int(i)]
        results.append({
            "score": float(score),
            "chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "text": c["text"][:600]
        })
    return {"question": req.question, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )

    #http://127.0.0.1:8000/docs 여기로 접속
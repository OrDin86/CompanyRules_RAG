# -*- coding: utf-8 -*-
import re, json
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config import PDF_DIR, INDEX_DIR, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, EMBED_MODEL_NAME

ARTICLE_RE = re.compile(r"(제\s*\d+\s*조\s*\([^)]+\)|제\s*\d+\s*조)")

def extract_text_per_page(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        txt = re.sub(r"\s+\n", "\n", txt).strip()
        pages.append({"page": i+1, "text": txt})
    return pages

def split_by_article(full_text: str):
    # 조항 헤더 기준 split (헤더 포함 유지)
    parts = ARTICLE_RE.split(full_text)
    if len(parts) == 1:
        return [full_text]

    chunks = []
    # parts: [pre, header1, body1, header2, body2, ...]
    pre = parts[0].strip()
    if pre:
        chunks.append(pre)

    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        merged = (header + "\n" + body).strip()
        chunks.append(merged)
        i += 2
    return chunks

def normalize_chunk(text: str):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def secondary_split(text: str, max_chars: int):
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    for pdf_path in tqdm(sorted(PDF_DIR.glob("*.pdf")), desc="PDF"):
        pages = extract_text_per_page(pdf_path)
        full_text = "\n".join([p["text"] for p in pages if p["text"]])

        # 1) 조항 단위 split
        article_chunks = split_by_article(full_text)

        # 2) 길이 보정 split
        final_chunks = []
        for ch in article_chunks:
            ch = normalize_chunk(ch)
            for sub in secondary_split(ch, CHUNK_MAX_CHARS):
                sub = normalize_chunk(sub)
                if len(sub) >= CHUNK_MIN_CHARS:
                    final_chunks.append(sub)

        for idx, ch in enumerate(final_chunks):
            all_chunks.append({
                "chunk_id": f"{pdf_path.stem}__{idx:04d}",
                "doc_title": pdf_path.stem,
                "text": ch
            })

    # 임베딩
    texts = ["query: " + c["text"] for c in all_chunks]  # e5 권장 포맷
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    # FAISS index
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 저장
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[OK] chunks={len(all_chunks)} saved to {INDEX_DIR}")

if __name__ == "__main__":
    build()
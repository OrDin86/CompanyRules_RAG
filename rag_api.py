# -*- coding: utf-8 -*-
import json
from pathlib import Path
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import INDEX_DIR, TOP_K, EMBED_MODEL_NAME, OPENAI_API_KEY, OPENAI_MODEL

app = FastAPI()
model = SentenceTransformer(EMBED_MODEL_NAME)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# load index + chunks
index = faiss.read_index(str(Path(INDEX_DIR) / "faiss.index"))
chunks = []
with open(Path(INDEX_DIR) / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

class QueryReq(BaseModel):
    question: str

class DraftReq(BaseModel):
    question: str

def retrieve_chunks(question: str, top_k: int = TOP_K):
    q = "query: " + question
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        c = chunks[int(i)]
        results.append({
            "score": float(score),
            "chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "text": c["text"]
        })
    return results

@app.post("/search")
def search(req: QueryReq):
    results = retrieve_chunks(req.question, TOP_K)
    return {"question": req.question, "results": results}

@app.post("/draft")
def draft(req: DraftReq):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되지 않았습니다.")

    retrieved = retrieve_chunks(req.question, TOP_K)

    context_blocks = []
    for i, r in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[근거 {i}]\n"
            f"문서명: {r['doc_title']}\n"
            f"청크ID: {r['chunk_id']}\n"
            f"유사도점수: {r['score']:.4f}\n"
            f"본문:\n{r['text']}\n"
        )

    context_text = "\n" + ("\n" + "="*80 + "\n").join(context_blocks)

    system_prompt = """
당신은 사내 규정 기반 질의응답 초안 작성 도우미다.

반드시 아래 규칙을 지켜라.
1. 제공된 근거 문서에 있는 내용만 사용한다.
2. 근거가 불충분하면 추정하지 말고 "규정 확인 필요"라고 쓴다.
3. 답변은 한국어로 작성한다.
4. 답변 형식은 아래를 따른다.

[답변 초안]
- 사용자에게 보여줄 자연스러운 답변

[근거]
- 문서명, 청크ID

[확인 필요]
- 근거가 약하거나 추가 확인이 필요한 항목
"""

    user_prompt = f"""
[질문]
{req.question}

[검색된 규정 근거]
{context_text}
"""

    # Responses API는 새 프로젝트에 권장되는 텍스트 생성 방식입니다.
    # 공식 문서 예시도 client.responses.create(...) 형태를 사용합니다.
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer_text = response.output_text

    return {
        "question": req.question,
        "draft_text": answer_text,
        "retrieved": [
            {
                "score": r["score"],
                "chunk_id": r["chunk_id"],
                "doc_title": r["doc_title"]
            }
            for r in retrieved
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
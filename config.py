# -*- coding: utf-8 -*-
from pathlib import Path
import os

# ====== CONFIG (여기만 수정하면 바로 실행) ======
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = Path(r"E:/RAG/rag_mvp/data/pdf")
INDEX_DIR = Path(r"E:/RAG/rag_mvp/data/index")

CHUNK_MAX_CHARS = 1200
CHUNK_MIN_CHARS = 200

# 임베딩 모델(로컬) - 설치가 쉬운 다국어 모델
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

# 검색 topK
TOP_K = 8

# ====== OpenAI ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5-mini"   # 빠르고 가벼운 초안용

# sk-proj-jrjeh2biwwwg6hfpggzqfmvlfhnjdvk7kk7dpsil9hum24me4dx0ybsnzfsvddlkdnk_kx9gzbt3blbkfjjme3bwciiuf_1vkeiot6tsgue3wqqbif7qwlxnytvxe9ghgcr6__eh_nggya8suukenmdh03qa
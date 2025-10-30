import numpy as np
from elasticsearch import Elasticsearch
import streamlit as st
from typing import List, Dict, Any
from utils import embed_text
from sentence_transformers import SentenceTransformer

# モデル読み込み（例：multilingual-e5）
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-large")

def search_documents(
    es_client: Elasticsearch,
    indices: List[str],
    keyword: str,
    start_year: int | None,
    end_year: int | None,
    jichitai: str | None,
    category1: str | None,
    category2: str | None,
    top_n: int = 10000
) -> List[Dict[str, Any]]:
    # フィルタ条件構築
    must_filters = []
    if start_year is not None:
        must_filters.append({"range": {"fiscal_year_start": {"gte": start_year}}})
    if end_year is not None:
        must_filters.append({"range": {"fiscal_year_end": {"lte": end_year}}})
    if jichitai:
        must_filters.append({"term": {"jichitai_name.keyword": jichitai}})
    if category1:
        must_filters.append({"term": {"category1.keyword": category1}})
    if category2:
        must_filters.append({"term": {"category2.keyword": category2}})

    query_body = {
        "size": top_n,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": keyword, "fields": ["title", "content"]}}
                ] + must_filters
            }
        }
    }

    # 検索実行
    response = es_client.search(index=indices, body=query_body, request_timeout=30)
    hits = response["hits"]["hits"]
    docs = [hit["_source"] for hit in hits]
    return docs

def rank_by_embedding(
    question: str,
    docs: List[Dict[str, Any]],
    top_k: int = 100
) -> List[Dict[str, Any]]:
    # 質問 embedding
    q_emb = embed_text(EMBED_MODEL, [question])[0]
    # ドキュメント embedding（都度）
    doc_texts = [doc["content"] for doc in docs]
    d_embs = embed_text(EMBED_MODEL, doc_texts)
    # コサイン類似度
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sims = [cosine(q_emb, d_emb) for d_emb in d_embs]
    ranked_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    ranked_docs = [docs[i] for i in ranked_idxs]
    return ranked_docs

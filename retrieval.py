# retrieval.py
import numpy as np
from elasticsearch import Elasticsearch
import streamlit as st
from typing import List, Dict, Any
import requests

def embed_text_gemini(
    texts: List[str],
    api_key: str,
    embedding_model: str = "gemini-embedding-001"
) -> np.ndarray:
    """
    Gemini API を使ってテキスト埋め込みを生成する関数。
    モデル「gemini-embedding-001」をデフォルトとしています。:contentReference[oaicite:1]{index=1}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{embedding_model}:embedContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    # API仕様に則り複数テキストを contents フィールドに入れる
    payload = {
        "model": embedding_model,
        "contents": [{"text": t} for t in texts]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # レスポンス形式： "embeddings": [ { "embedding": [...] }, ... ]
    embeddings = [item["embedding"] for item in data["embeddings"]]
    return np.array(embeddings)

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
    """
    Elasticsearch に対してキーワード＋フィルタ条件を用いた検索を行い、
    最大 top_n 件を取得します。
    """
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
                    {"multi_match": {"query": keyword, "fields": ["title", "content_text"]}}
                ] + must_filters
            }
        }
    }

    response = es_client.search(index=indices, body=query_body, request_timeout=30)
    hits = response["hits"]["hits"]
    docs = [hit["_source"] for hit in hits]
    return docs

def rank_by_embedding(
    question: str,
    docs: List[Dict[str, Any]],
    api_key: str,
    embedding_model: str = "gemini-embedding-001",
    top_k: int = 100
) -> List[Dict[str, Any]]:
    """
    質問に対して embedding を生成し、docs に対しても embedding を生成して
    コサイン類似度により上位 top_k 件を抽出します。
    """
    # 質問 embedding
    q_emb = embed_text_gemini([question], api_key, embedding_model)[0]
    # docs の内容テキスト抽出
    doc_texts = [doc["content"] for doc in docs]
    # docs embedding（都度生成）
    d_embs = embed_text_gemini(doc_texts, api_key, embedding_model)
    # コサイン類似度計算
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sims = [cosine(q_emb, d_emb) for d_emb in d_embs]
    # 上位 top_k のインデックス取得
    ranked_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    ranked_docs = [docs[i] for i in ranked_idxs]
    return ranked_docs

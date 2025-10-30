# retrieval.py
import json
import numpy as np
from elasticsearch import Elasticsearch
import streamlit as st
from typing import List, Dict, Any
import requests

# =====================================================
# ============ Gemini Embedding 関数 ===================
# =====================================================

def embed_text_gemini(
    texts: List[str],
    api_key: str,
    embedding_model: str = "gemini-embedding-001"
) -> np.ndarray:
    """
    Gemini API を使ってテキスト埋め込みを生成する関数。
    モデル「gemini-embedding-001」をデフォルトとしています。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{embedding_model}:embedContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "model": embedding_model,
        "contents": [{"text": t} for t in texts]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # レスポンス形式："embeddings": [{"embedding": [...]}, ...]
        embeddings = [item["embedding"] for item in data["embeddings"]]
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Gemini埋め込み生成でエラーが発生しました: {e}")
        return np.zeros((len(texts), 1))


# =====================================================
# ============ Elasticsearch 検索処理 =================
# =====================================================

def build_search_query(filters: Dict[str, Any]) -> dict:
    """
    sidebar_filters() から受け取ったフィルタ条件を元に Elasticsearch クエリを構築
    """
    must_clauses = []
    should_clauses = []
    must_not_clauses = []
    filter_clauses = []

    # AND条件
    for w in [w.strip() for w in filters["and_input"].replace("　", " ").split() if w.strip()]:
        if filters["include_title"]:
            must_clauses.append({
                "bool": {
                    "should": [
                        {"match_phrase": {"content_text": w}},
                        {"match_phrase": {"title": w}}
                    ],
                    "minimum_should_match": 1
                }
            })
        else:
            must_clauses.append({"match_phrase": {"content_text": w}})

    # OR条件
    for w in [w.strip() for w in filters["or_input"].replace("　", " ").split() if w.strip()]:
        if filters["include_title"]:
            should_clauses.append({
                "bool": {
                    "should": [
                        {"match_phrase": {"content_text": w}},
                        {"match_phrase": {"title": w}}
                    ],
                    "minimum_should_match": 1
                }
            })
        else:
            should_clauses.append({"match_phrase": {"content_text": w}})

    # NOT条件
    for w in [w.strip() for w in filters["not_input"].replace("　", " ").split() if w.strip()]:
        if filters["include_title"]:
            must_not_clauses.append({
                "bool": {
                    "should": [
                        {"match_phrase": {"content_text": w}},
                        {"match_phrase": {"title": w}}
                    ],
                    "minimum_should_match": 1
                }
            })
        else:
            must_not_clauses.append({"match_phrase": {"content_text": w}})

    # 年度
    if filters["years"]:
        year_should = []
        for y in filters["years"]:
            cond_between = {
                "bool": {
                    "must": [
                        {"range": {"fiscal_year_start": {"lte": y}}},
                        {"range": {"fiscal_year_end": {"gte": y}}}
                    ]
                }
            }
            cond_start_eq_when_no_end = {
                "bool": {
                    "must": [{"term": {"fiscal_year_start": y}}],
                    "must_not": [{"exists": {"field": "fiscal_year_end"}}]
                }
            }
            year_should.append(cond_between)
            year_should.append(cond_start_eq_when_no_end)

        filter_clauses.append({
            "bool": {
                "should": year_should,
                "minimum_should_match": 1
            }
        })

    # 自治体コード
    if filters["codes"]:
        filter_clauses.append({"terms": {"code": filters["codes"]}})

    # カテゴリ
    if filters["categories"]:
        filter_clauses.append({"terms": {"category": filters["categories"]}})

    query = {"bool": {}}
    if must_clauses:
        query["bool"]["must"] = must_clauses
    if should_clauses:
        query["bool"]["should"] = should_clauses
        query["bool"]["minimum_should_match"] = 1
    if must_not_clauses:
        query["bool"]["must_not"] = must_not_clauses
    if filter_clauses:
        query["bool"]["filter"] = filter_clauses

    if not query["bool"]:
        return {"match_all": {}}
    return query


def search_documents(es_client: Elasticsearch, indices: List[str], query: dict, limit: int = 10000) -> List[Dict[str, Any]]:
    """
    Elasticsearch で指定クエリを実行し、最大 limit 件を返す。
    """
    body = {
        "size": limit,
        "query": query
    }
    res = es_client.search(index=indices, body=body, request_timeout=30)
    hits = res.get("hits", {}).get("hits", [])
    docs = [h["_source"] for h in hits]
    return docs


# =====================================================
# ============ Gemini Embeddingによる類似度算出 ========
# =====================================================

def rank_by_embedding(
    question: str,
    docs: List[Dict[str, Any]],
    api_key: str,
    embedding_model: str = "gemini-embedding-001",
    top_k: int = 100
) -> List[Dict[str, Any]]:
    """
    質問に対して Gemini Embedding を生成し、
    docs に対しても Embedding を生成してコサイン類似度で上位 top_k を抽出。
    """
    if not docs:
        return []

    # --- 質問の埋め込み ---
    q_emb = embed_text_gemini([question], api_key, embedding_model)[0]

    # --- ドキュメント埋め込み ---
    doc_texts = [doc.get("content_text", "") for doc in docs]
    d_embs = embed_text_gemini(doc_texts, api_key, embedding_model)

    # --- 類似度算出 ---
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine(q_emb, d_emb) for d_emb in d_embs]
    ranked_idxs = np.argsort(sims)[::-1][:top_k]
    ranked_docs = [docs[i] for i in ranked_idxs]

    return ranked_docs

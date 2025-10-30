import streamlit as st
import requests
from typing import List, Dict

def generate_answer(
    question: str,
    docs: List[Dict],
    api_key: str,
    model: str = "gemini-2.0-flash",
    max_output_tokens: int = 500
) -> str:
    # プロンプト構築
    context = ""
    for doc in docs:
        context += f"資料名: {doc['title']}\nURL: {doc['url']}\n内容要約: {doc.get('summary', '')}\n---\n"
    prompt = f"""以下の資料を参照して、ユーザーの質問に日本語で回答してください。

{context}
質問: {question}

回答を記述してください。回答の最後に「参考資料：」として、資料名とURLを列挙してください。"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateText".format(model=model)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": {
            "text": prompt
        },
        "maxOutputTokens": max_output_tokens,
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    answer = data["candidates"][0]["output"]  # 実レスポンス形式は確認要
    return answer

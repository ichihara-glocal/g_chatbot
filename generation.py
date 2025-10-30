import streamlit as st
from typing import List, Dict
import openai  # もしくはGemini API用別クライアント

def generate_answer(
    question: str,
    docs: List[Dict],
    api_key: str,
    max_chars: int = 1000
) -> str:
    openai.api_key = api_key
    # 要約文作成
    context = ""
    for doc in docs:
        context += f"資料名: {doc['title']}\nURL: {doc['url']}\n内容要約: {doc.get('summary', '')}\n---\n"

    prompt = f"""以下の資料を参照して、ユーザーの質問に日本語で回答してください。

{context}

質問: {question}

回答を記述してください。回答の最後に「参考資料：」として、資料名とURLを列挙してください。
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 仮
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    return answer

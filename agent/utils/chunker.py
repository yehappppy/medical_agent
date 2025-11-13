import os
import httpx
import requests
from pydantic import SecretStr
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings


def get_embedding_model(
        model_name: str = "Qwen/Qwen3-Embedding-8B") -> OpenAIEmbeddings:

    embedding_model = OpenAIEmbeddings(
        model=model_name,
        base_url=os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1"),
        api_key=SecretStr(os.getenv("API_KEY", "")),
    )
    return embedding_model


def sync_rerank(
    query: str,
    documents: List[str],
) -> Dict[str, Any]:
    url = f"{os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")}/rerank"

    payload = {
        "model": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
        "query": query,
        "documents": documents,
        "top_n": int(os.getenv("RERANK_TOP_N", 5)),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("API_KEY", "")}"
    }
    print(headers)
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()


async def async_rerank(
    query: str,
    documents: List[str],
) -> Dict[str, Any]:
    url = f"{os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")}/rerank"

    payload = {
        "model": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
        "query": query,
        "documents": documents,
        "top_n": int(os.getenv("RERANK_TOP_N", 5)),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("API_KEY", "")}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

    return response.json()

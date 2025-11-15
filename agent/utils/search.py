import os
import requests
import asyncio
from loguru import logger
from typing import List, Dict, Any
from .nrag import AsyncQdrantRAG

qdrant_client = AsyncQdrantRAG()
mode = os.getenv("MODE", "dev")
rag = os.getenv("RAG", "nrag").upper()
summary_collection_name = os.getenv("QDRANT_COLLECTION",
                                    "medical_document_summaries")


async def retrieve(query: str) -> List[Dict]:
    summary_results = await qdrant_client.search(summary_collection_name,
                                                 query)
    tasks = []
    for summary_result in summary_results:
        metadata: dict = summary_result.get("metadata", {})
        file_stem = metadata.get("file_stem", None)
        tasks.append(qdrant_client.search(f"{rag}_{file_stem}_{mode}", query))

    all_results = []
    chunk_results_list = await asyncio.gather(*tasks, return_exceptions=True)

    for chunk_results in chunk_results_list:
        if isinstance(chunk_results, Exception):
            logger.error(f"Error in chunk search: {chunk_results}")
            continue
        elif isinstance(chunk_results, List):
            all_results.extend(chunk_results)
    logger.info(
        f"Successfully retrieved {len(all_results)} chunks for query {query}!")
    return all_results


def rerank(
    query: str,
    documents: List[Dict],
) -> List[Dict]:
    url = f"{os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")}/rerank"

    payload = {
        "model": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
        "query": query,
        "documents": [document["content"] for document in documents],
        "top_n": int(os.getenv("RERANK_TOP_N", 5)),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("API_KEY", "")}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        logger.info(f"Reranking error: {response.text}")
        response.raise_for_status()
    reponse_dict: dict = response.json()
    results: List[dict] = reponse_dict.get("results", [])
    reranked_docs = [documents[result["index"]] for result in results]
    return reranked_docs

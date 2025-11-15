import os
import asyncio
from loguru import logger
from typing import List, Dict
from agent.utils.nrag import AsyncQdrantRAG

qdrant_client = AsyncQdrantRAG()
mode = os.getenv("MODE", "dev")
rag = os.getenv("RAG", "nrag").upper()
summary_collection_name = os.getenv("QDRANT_COLLECTION",
                                    "medical_document_summaries")


async def search(query: str) -> List[Dict]:
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
    logger.info(f"Successfully retrieved {len(all_results)} chunks.")
    return all_results

import os
import json
import hashlib
from loguru import logger
from typing import List, Dict
from agent.utils.chunker import get_embedding_model

from qdrant_client.http import models
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams


class AsyncQdrantRAG:

    def __init__(self):
        self.client = AsyncQdrantClient(host=os.getenv("QDRANT_HOST",
                                                       "localhost"),
                                        port=int(os.getenv(
                                            "QDRANT_PORT", 6333)))
        self.vector_size = int(os.getenv("QDRENT_VECTOR_SIZE", 4096))
        self.distance = getattr(Distance, os.getenv("QDRANT_DISTANCE",
                                                    "COSINE"))
        self.embedding_model = get_embedding_model()

    def generate_qdrant_id(self, document: Dict) -> str:
        content_to_hash = document.get("content", "") + json.dumps(
            document.get("metadata", {}), sort_keys=True)
        content_hash = hashlib.sha256(
            content_to_hash.encode('utf-8')).hexdigest()
        logger.debug(f"Generated Qdrant ID: {content_hash} for document")
        return content_hash[:32]

    async def create_collection(self, collection_name: str):
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.vector_size,
                                                distance=self.distance))
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    async def add_documents(
        self,
        documents: List[Dict],
        collection_name: str,
    ):
        points = []
        for document in documents:
            embedding = self.embedding_model.embed_query(
                document.get("content", ""))
            point = models.PointStruct(id=self.generate_qdrant_id(document),
                                       vector=embedding,
                                       payload={
                                           "content":
                                           document.get("content", ""),
                                           "metadata":
                                           document.get("metadata", {}),
                                       })
            points.append(point)

        await self.client.upsert(collection_name=collection_name,
                                 points=points)
        logger.info(f"Added {len(documents)} documents to collection")

    async def search(
        self,
        collection_name: str,
        query: str,
    ) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=int(os.getenv("QDRANT_TOP_K", 5)),
                with_payload=True)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id":
                    result.id,
                    "content":
                    result.payload.get("content", ""),
                    "metadata":
                    result.payload.get("metadata", {}),
                    "score":
                    result.score,
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def delete_collection(self, collection_name: str):
        try:
            await self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    async def close(self):
        """Close the client connection"""
        await self.client.close()

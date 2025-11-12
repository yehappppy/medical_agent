import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Optional, Union
import asyncio
from loguru import logger
from pathlib import Path


class AsyncQdrantRAG:

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 vector_size: int = 512,
                 collection_name: str = "medical_knowledge_base",
                 distance: str = "Cosine"):
        """
        Initialize async Qdrant client for RAG
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Size of embedding vectors
            distance: Distance metric (Cosine, Euclid, Dot)
        """
        self.client = AsyncQdrantClient(host=os.getenv("QDRANT_HOST",
                                                       "localhost"),
                                        port=int(os.getenv(
                                            "QDRANT_PORT", 6333)))
        self.vector_size = int(os.getenv("QDRANT_DIMENSION", vector_size))
        self.collection_name = collection_name
        self.distance = getattr(Distance, distance.upper())

    async def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size,
                                                distance=self.distance))
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(
                    f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    async def add_documents(self,
                            documents: List[Dict],
                            embeddings: List[List[float]],
                            ids: Optional[List[Union[int, str]]] = None):
        """
        Add documents with embeddings to collection

        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            embeddings: List of embedding vectors
            ids: Optional list of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings")

        if ids is None:
            ids = list(range(len(documents)))

        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = models.PointStruct(id=str(ids[i]),
                                       vector=embedding,
                                       payload={
                                           "content": doc.get("content", ""),
                                           "metadata": doc.get("metadata", {}),
                                           "source":
                                           doc.get("source", "unknown")
                                       })
            points.append(point)

        await self.client.upsert(collection_name=self.collection_name,
                                 points=points)
        logger.info(f"Added {len(documents)} documents to collection")

    async def search(self,
                     query_embedding: List[float],
                     top_k: int = 5,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filter conditions
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)))
                if conditions:
                    qdrant_filter = models.Filter(must=conditions)

            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True)

            # Format results
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
                    "source":
                    result.payload.get("source", "unknown")
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def delete_collection(self):
        """Delete the collection"""
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    async def close(self):
        """Close the client connection"""
        await self.client.aclose()

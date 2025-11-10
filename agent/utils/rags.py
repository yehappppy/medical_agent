from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from typing import List, Dict, Any
import uuid


class QdrantRAGClient:

    def __init__(self,
                 host="localhost",
                 port=6333,
                 collection_name="documents"):
        """
        Initialize Qdrant RAG client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self, vector_size=384):
        """
        Create Qdrant collection with specified vector size
        """
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(collection_name=self.collection_name,
                                          vectors_config=models.VectorParams(
                                              size=vector_size,
                                              distance=models.Distance.COSINE))

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the collection
        
        Args:
            documents: List of dicts with 'text' and optional metadata
        """
        points = []
        for doc in documents:
            vector = self.encoder.encode(doc['text']).tolist()
            point = PointStruct(id=str(uuid.uuid4()),
                                vector=vector,
                                payload={
                                    "text": doc['text'],
                                    "metadata": doc.get('metadata', {})
                                })
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of documents with scores
        """
        query_vector = self.encoder.encode(query).tolist()

        results = self.client.search(collection_name=self.collection_name,
                                     query_vector=query_vector,
                                     limit=top_k)

        return [{
            "text": result.payload["text"],
            "metadata": result.payload["metadata"],
            "score": result.score
        } for result in results]

    def delete_collection(self):
        """Delete the collection (use with caution)"""
        self.client.delete_collection(self.collection_name)

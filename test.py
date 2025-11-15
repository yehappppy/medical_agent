import os
import asyncio
from agent.utils import (
    logger,
    get_agent,
    get_embedding_model,
    image_to_base64,
    rerank,
    AsyncQdrantRAG,
)

EMBEDDING_MODEL_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
}

MK_AGENT = get_agent(name="SLM")
LLM_AGENT = get_agent(name="LLM", tags=["stream"])
VLM_AGENT = get_agent(name="VLM", tags=["stream"])
OCR_AGENT = get_agent(name="OCR")

logger.info("Agent initialized successfully.")

MK_AGENT_response = MK_AGENT.invoke([{
    "role":
    "user",
    "content":
    "Hello, Medical Knowledge Agent!"
}]).content
logger.info(f"MK_AGENT response: {MK_AGENT_response}")

LLM_AGENT_response = LLM_AGENT.invoke([{
    "role":
    "user",
    "content":
    "Hello, Language Model Agent!"
}]).content
logger.info(f"LLM_AGENT response: {LLM_AGENT_response}")

VLM_AGENT_response = VLM_AGENT.invoke([{
    "role":
    "user",
    "content": [{
        "type": "text",
        "text": "Describe the image."
    }, {
        "type": "image_url",
        "image_url": {
            "url": "https://http.cat/images/200.jpg"
        }
    }]
}]).content
logger.info(f"VLM_AGENT response: {VLM_AGENT_response}")

image_base64 = image_to_base64("data/test.jpg")
OCR_AGENT_response = OCR_AGENT.invoke([{
    "role":
    "user",
    "content": [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }
    }, {
        "type": "text",
        "text": "<image>\n<|grounding|>OCR this image."
    }]
}]).content
logger.info(f"OCR_AGENT response: {OCR_AGENT_response}")

embedding_model = get_embedding_model()
text = "LangChain is the framework for building context-aware reasoning applications"
single_vector = embedding_model.embed_query(text)
logger.info(f"Single vector embedding for the text: {single_vector[:5]}...")

rerank_result = rerank(
    query="Python programming",
    documents=[
        {
            "content": "Python is great",
            "metadata": {
                "file_name": "doc1"
            }
        },
        {
            "content": "Java is fast",
            "metadata": {
                "file_name": "doc2"
            }
        },
        {
            "content": "C++ is powerful",
            "metadata": {
                "file_name": "doc1"
            }
        },
    ],
)
logger.info(rerank_result)

qdrant_client = AsyncQdrantRAG()
collection_name = os.getenv("QDRANT_COLLECTION", "medical_document_summaries")
asyncio.run(qdrant_client.create_collection(collection_name=collection_name))
asyncio.run(
    qdrant_client.add_documents(
        documents=[
            {
                "content": "1 + 1 = 2",
                "metadata": {
                    "source": "doc1.pdf"
                }
            },
            {
                "content": "2 + 2 = 4",
                "metadata": {
                    "source": "doc2.pdf"
                }
            },
        ],
        collection_name=collection_name,
    ))
result = asyncio.run(
    qdrant_client.search(collection_name=collection_name, query="1 + 1 = ?"))
logger.info(result)
logger.info("Test completed.")

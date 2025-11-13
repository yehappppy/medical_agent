import os
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


def get_embedding_model(
        model_name: str = "Qwen/Qwen3-Embedding-8B") -> OpenAIEmbeddings:

    embedding_model = OpenAIEmbeddings(
        model=model_name,
        base_url=os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1"),
        api_key=SecretStr(os.getenv("API_KEY", "")),
    )
    return embedding_model

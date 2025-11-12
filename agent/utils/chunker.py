from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


def get_embedding_model(base_url: str, api_key: SecretStr,
                        model_name: str) -> OpenAIEmbeddings:
    embedding_model = OpenAIEmbeddings(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    return embedding_model

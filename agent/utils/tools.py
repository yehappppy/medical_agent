import os
import io
import sys
import base64
from PIL import Image
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Union, List
from pydantic import SecretStr
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_agent(name: str, tags: list[str] | None = None) -> ChatOpenAI:
    base_url = os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1")
    api_key = SecretStr(os.getenv("API_KEY", ""))
    model_name = os.getenv(f"{name}_MODEL", "")
    temperature = float(os.getenv(f"{name}_TEMPERATURE", 0.3))
    max_completion_tokens = int(os.getenv(f"{name}_MAX_TOKENS", 16384))
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        tags=tags,
    )
    return model


def get_embedding_model(
        model_name: str = "Qwen/Qwen3-Embedding-8B") -> OpenAIEmbeddings:

    embedding_model = OpenAIEmbeddings(
        model=model_name,
        base_url=os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1"),
        api_key=SecretStr(os.getenv("API_KEY", "")),
    )
    return embedding_model


def get_logger(module_name: str = "medical_agent"):
    """Setup logger with module-specific log files"""
    logger.remove()

    logs_dir = Path("log")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{module_name}_{timestamp}.log"

    logger.add(
        sys.stdout,
        format=
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True)

    logger.add(
        log_filename,
        format=
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip")

    return logger


def image_to_base64(
    image_input: Union[str, Image.Image, List[Image.Image]]
) -> Union[str, List[str]]:
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded = base64.b64encode(binary_data).decode('utf-8')
            return base64_encoded

    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        return base64_encoded

    elif isinstance(image_input, list):
        base64_list = []
        for img in image_input:
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"List item must be a PIL Image object, got {type(img)}")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_list.append(base64_str)
        return base64_list

    else:
        raise TypeError(
            f"Input must be str, PIL.Image.Image, or List[PIL.Image.Image], got {type(image_input)}"
        )


def pdf_to_image_list(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

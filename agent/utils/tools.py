import os
import sys
from loguru import logger
from pathlib import Path
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from datetime import datetime


# Initialize ChatOpenAI model
def get_agent(tag: str = "stream") -> ChatOpenAI:
    BASE_URL = os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1")
    API_KEY = os.getenv("API_KEY", None)
    if not API_KEY:
        raise ValueError("API_KEY environment variable is not set.")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-235B-A22B-Instruct")
    MAX_COMPLETION_TOKENS = int(os.getenv("MAX_TOKENS", "16384"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_P = float(os.getenv("TOP_P", "0.7"))
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
        tags=["stream"] if tag == "stream" else None,
    )
    return model


def get_logger(module_name: str = "medical_agent") -> logger:
    """Setup logger with module-specific log files"""
    logger.remove()

    # Create logs directory
    logs_dir = Path("log")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamped filename with module name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{module_name}_{timestamp}.log"

    # Console handler
    logger.add(
        sys.stdout,
        format=
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True)

    # File handler with module name + timestamp
    logger.add(
        log_filename,
        format=
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        level="INFO",
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip"  # Compress old logs
    )

    return logger


logger = get_logger()

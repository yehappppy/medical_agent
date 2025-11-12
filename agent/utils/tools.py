import sys
from loguru import logger
from pathlib import Path
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from datetime import datetime


# Initialize ChatOpenAI model
def get_agent(base_url: str,
              api_key: SecretStr,
              model_name: str,
              temperature: float = 0.3,
              top_p: float = 0.7,
              max_completion_tokens: int = 16384,
              tag: str = "stream") -> ChatOpenAI:
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        base_url=base_url,
        api_key=api_key,
        tags=[tag],
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

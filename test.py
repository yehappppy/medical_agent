import os
from agent.utils.tools import logger, get_agent

API_CONFIG = {
    "base_url": os.getenv("MODEL_URL", "https://api.siliconflow.cn/v1"),
    "api_key": os.getenv("API_KEY", None),
}

LLM_CONFIG = {
    "model_name": os.getenv("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    "max_completion_tokens": int(os.getenv("LLM_MAX_TOKENS", 16384)),
    "temperature": float(os.getenv("LLM_TEMPERATURE", 0.3)),
    "top_p": float(os.getenv("LLM_TOP_P", 0.7)),
}

VLM_CONFIG = {
    "model_name": os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
    "max_completion_tokens": int(os.getenv("VLM_MAX_TOKENS", 16384)),
    "temperature": float(os.getenv("VLM_TEMPERATURE", 0.3)),
    "top_p": float(os.getenv("VLM_TOP_P", 0.7)),
}

SLM_CONFIG = {
    "model_name": os.getenv("SLM_MODEL", "Qwen/Qwen3-14B"),
    "max_completion_tokens": int(os.getenv("SLM_MAX_TOKENS", 4096)),
    "temperature": float(os.getenv("SLM_TEMPERATURE", 0.1)),
}

EMBEDDING_MODEL_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
}

MK_AGENT = get_agent(**API_CONFIG, **SLM_CONFIG, tag="MK")
LLM_AGENT = get_agent(**API_CONFIG, **LLM_CONFIG, tag="LLM")
VLM_AGENT = get_agent(**API_CONFIG, **VLM_CONFIG, tag="VLM")

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

logger.info("Test completed.")

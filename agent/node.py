import os
import base64
from io import BytesIO
from typing import Dict, Any, List
from PIL import Image
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path
from agent.state import InputProcessorState, SymptomFinderState, SymptomCheckerState, MedicalKnowledgeState
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


######################################################################
#################### Medical Knowledge Retriever #####################
######################################################################
def KnowledgeRetriever(state: MedicalKnowledgeState):
    pass
    update = {
        "references": "Medical knowledge retrieval is not yet implemented.",
    }
    return update


def KnowledgeReasoner(state: MedicalKnowledgeState):
    pass
    update = {
        "call_rag": True,
        "query": "Medical knowledge reasoning is not yet implemented.",
    }
    return update


def KnowledgeAgentInternalRouter(state: MedicalKnowledgeState):
    if True:
        return "Continue RAG"
    return "Exit RAG"


def MedicalAgentDiagnosisRouter(state: MedicalKnowledgeState):
    if True:
        return "Need Clinical Records"
    return "No Need"


def MedicalAgentVerdictRouter(state: MedicalKnowledgeState):
    if True:
        return "Need Medical Knowledge"
    return "No Need"


def KnowledgeAgentOutputRouter(state: MedicalKnowledgeState):
    if 1:
        return "Medical Knowledge"
    else:
        return "elevant Clinical Cases"


######################################################################
########################### Medical Agent ############################
######################################################################
def InputProcessor(state: InputProcessorState) -> dict:
    """Convert image to base64 string."""
    medical_recode_path: Path = Path(state["medical_recode_path"])
    medical_image_path: Path = Path(state["medical_image_path"])
    with open(medical_image_path, "rb") as img_file:
        medical_image = base64.b64encode(img_file.read()).decode("utf-8")
    # TODO: OCR for medical_recode_path if it's an image/PDF
    medical_recode: str = ""
    update: dict = {
        "medical_recode": medical_recode,
        "medical_image": medical_image
    }
    return update


def ImageClassifier(state: SymptomFinderState):
    pass
    update = {
        "reasoning": "Medical image classification is not yet implemented.",
        "symptom": "Medical image classification is not yet implemented.",
        "disease": "Medical image classification is not yet implemented."
    }
    return update


def SymptomFinder(state: SymptomFinderState):
    pass
    update = {
        "symptom": "Medical symptom finding is not yet implemented.",
    }
    return update


def HumanReviewer(state: SymptomFinderState):
    pass
    update = {
        "symptom": "Medical symptom finding is not yet implemented.",
    }
    return update


def SymptomChecker(state: SymptomCheckerState):
    pass
    update = {
        "self_reflection": "Medical symptom checking is not yet implemented.",
    }
    return update


def ReportGenerator(state: SymptomCheckerState):
    pass
    update = {
        "medical_report": "Medical symptom checking is not yet implemented.",
    }
    return update

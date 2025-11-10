import base64
from io import BytesIO
from typing import Dict, Any, List
from PIL import Image
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path
from agent.state import InputProcessorState, SymptomFinderState, SymptomCheckerState, MedicalKnowledgeState


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

from typing import TypedDict, List, Dict, Any, Optional


class MedicalAgentInputState(TypedDict):
    medical_image_path: str
    medical_recode_path: str


class MedicalOutputInputState(TypedDict):
    medical_report: str


class InputProcessorState(TypedDict):
    medical_image_path: str
    medical_recode_path: str
    medical_recode: str
    medical_image: str


class SymptomFinderState(TypedDict):
    medical_image: str
    reasoning: str
    symptom: str
    disease: str
    bb: Optional[List[int]]


class SymptomCheckerState(TypedDict):
    symptom: str
    disease: str
    medical_report: str


class MedicalKnowledgeState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]

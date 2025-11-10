# TODO: this is a demo workflow, please modify it according to your needs
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from agent.state import (MedicalAgentInputState, MedicalKnowledgeState)
from agent.node import (KnowledgeRetriever, KnowledgeReasoner, InputProcessor,
                        ImageClassifier, SymptomFinder, HumanReviewer,
                        SymptomChecker, ReportGenerator,
                        KnowledgeAgentInternalRouter,
                        MedicalAgentDiagnosisRouter, MedicalAgentVerdictRouter,
                        KnowledgeAgentOutputRouter)

######################################################################
#################### Medical Knowledge Retriever #####################
######################################################################

MedicalKnowledgeRetriever = StateGraph(MedicalKnowledgeState)
MedicalKnowledgeRetriever.add_node("Knowledge Retrieval", KnowledgeRetriever)

MedicalKnowledgeRetriever.add_node("Knowledge Reasoning", KnowledgeReasoner)

MedicalKnowledgeRetriever.add_edge(START, "Knowledge Retrieval")
MedicalKnowledgeRetriever.add_edge("Knowledge Retrieval",
                                   "Knowledge Reasoning")
MedicalKnowledgeRetriever.add_conditional_edges(
    "Knowledge Reasoning", KnowledgeAgentInternalRouter, {
        "Continue RAG": "Knowledge Retrieval",
        "Exit RAG": END
    })
MedicalKnowledgeRetriever.add_edge("Knowledge Retrieval", END)

MedicalKnowledgeRetriever = MedicalKnowledgeRetriever.compile()

######################################################################
########################### Medical Agent ############################
######################################################################

app = StateGraph(MedicalAgentInputState)
app.add_node("Input Processor", InputProcessor)
app.add_node("Medical Knowledge Agent", MedicalKnowledgeRetriever)
app.add_node("Image Classifier", ImageClassifier)
app.add_node("Symptom Finder", SymptomFinder)
app.add_node("Human Reviewer", HumanReviewer)
app.add_node("Symptom Checker", SymptomChecker)
app.add_node("Report Generator", ReportGenerator)

app.add_edge(START, "Input Processor")
app.add_conditional_edges(
    "Input Processor", MedicalAgentDiagnosisRouter, {
        "Need Medical Knowledge": "Medical Knowledge Agent",
        "No Need": "Image Classifier"
    })
app.add_edge("Image Classifier", "Symptom Finder")
app.add_conditional_edges(
    "Symptom Finder", MedicalAgentVerdictRouter, {
        "Need Clinical Records": "Medical Knowledge Agent",
        "No Need": "Symptom Checker"
    })
app.add_conditional_edges("Medical Knowledge Agent",
                          KnowledgeAgentOutputRouter, {
                              "Medical Knowledge": "Image Classifier",
                              "Clinical Records": "Symptom Checker"
                          })
app.add_edge("Symptom Checker", "Human Reviewer")
app.add_edge("Human Reviewer", "Report Generator")
app.add_edge("Report Generator", END)

app = app.compile()
print(app.get_graph(xray=True).draw_mermaid())

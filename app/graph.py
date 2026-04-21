"""
LangGraph Orchestration — Medical AI Pipeline

Change from previous version:
- State now carries structured dicts (findings, answer) instead of raw strings
- vlm_node returns dict, llm_node receives dict → cleaner data contract
- No other logic changes
"""

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.llm import generate_answer
from app.rag import retrieve_context
from app.vlm import extract_findings

logger = logging.getLogger(__name__)


class MedicalState(TypedDict):
    question: str
    image_bytes: Optional[bytes]
    findings: Optional[dict]    # VLMOutput dict: {study_type, anatomy, findings, impression}
    context: Optional[str]      # RAG retrieved text
    answer: Optional[dict]      # LLMOutput dict: {summary, context_used, answer, red_flags, recommendation}


def vlm_node(state: MedicalState) -> MedicalState:
    """Extract structured image findings via MedGemma multimodal."""
    logger.info("▶ vlm_node")
    image_bytes = state.get("image_bytes")
    findings = extract_findings(image_bytes) if image_bytes else {
        "study_type": "None", "anatomy": "None",
        "findings": "No image provided.", "impression": "N/A"
    }
    return {**state, "findings": findings}


def rag_node(state: MedicalState) -> MedicalState:
    """Retrieve medical context using findings + question as combined query."""
    logger.info("▶ rag_node")
    findings = state.get("findings", {})
    combined_query = f"{state['question']} {findings.get('findings', '')}".strip()
    context = retrieve_context(combined_query)
    return {**state, "context": context}


def llm_node(state: MedicalState) -> MedicalState:
    """Generate structured answer via MedGemma text mode."""
    logger.info("▶ llm_node")
    answer = generate_answer(
        question=state["question"],
        findings=state.get("findings", {}),
        context=state.get("context", "No context retrieved."),
    )
    return {**state, "answer": answer}


def build_graph() -> StateGraph:
    graph = StateGraph(MedicalState)
    graph.add_node("vlm", vlm_node)
    graph.add_node("rag", rag_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("vlm")
    graph.add_edge("vlm", "rag")
    graph.add_edge("rag", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


pipeline = build_graph()
logger.info("✅ LangGraph pipeline compiled.")

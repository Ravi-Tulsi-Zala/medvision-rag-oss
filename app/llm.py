"""
LLM Module — MedGemma via Vertex AI (findings + context → structured answer)
Thin wrapper: formats prompt, calls API (text-only), parses JSON.
"""

import json
import logging
import re

from pydantic import BaseModel
from app.medgemma import MAX_TOKENS_LLM, call_medgemma
from app.prompt import LLM_STRUCTURED_PROMPT, MEDICAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMOutput(BaseModel):
    summary:        str       = "No summary available."
    context_used:   str       = "No context applied."
    answer:         str       = "No answer generated."
    red_flags:      list[str] = []
    recommendation: str       = "Please consult a licensed healthcare professional."


def _parse_llm_json(raw: str) -> LLMOutput:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data.get("red_flags"), str):
                data["red_flags"] = [data["red_flags"]]
            return LLMOutput(**{k: v for k, v in data.items() if k in LLMOutput.model_fields})
        except Exception as e:
            logger.warning(f"LLM JSON parse failed: {e}")
    return LLMOutput(answer=raw.strip() or "No answer generated.")


def generate_answer(question: str, findings: dict, context: str) -> dict:
    """Call MedGemma text-only on Vertex AI and return structured answer dict."""
    findings_text = (
        f"Study: {findings.get('study_type', 'Unknown')}\n"
        f"Anatomy: {findings.get('anatomy', 'Unknown')}\n"
        f"Findings: {findings.get('findings', 'None')}\n"
        f"Impression: {findings.get('impression', 'None')}"
    )
    prompt = f"{MEDICAL_SYSTEM_PROMPT}\n\n" + LLM_STRUCTURED_PROMPT.format(
        findings=findings_text,
        context=context,
        question=question,
    )

    raw = call_medgemma(prompt=prompt, image_bytes=None, max_tokens=MAX_TOKENS_LLM)
    logger.info(f"LLM raw: {raw[:120]}")
    return _parse_llm_json(raw).model_dump()

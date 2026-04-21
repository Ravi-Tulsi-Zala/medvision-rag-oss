"""
VLM Module — MedGemma via Vertex AI (image → structured findings)
Now a thin wrapper: formats prompt, calls API, parses JSON response.
No model weights, no GPU, no downloads.
"""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel
from app.medgemma import MAX_TOKENS_VLM, call_medgemma
from app.prompt import VLM_STRUCTURED_PROMPT

logger = logging.getLogger(__name__)


class VLMOutput(BaseModel):
    study_type: str = "Unknown"
    anatomy:    str = "Not identified"
    findings:   str = "No findings extracted."
    impression: str = "Unable to determine."


def _parse_vlm_json(raw: str) -> VLMOutput:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return VLMOutput(**{k: v for k, v in data.items() if k in VLMOutput.model_fields})
        except Exception as e:
            logger.warning(f"VLM JSON parse failed: {e}")
    return VLMOutput(findings=raw.strip() or "No findings extracted.")


def extract_findings(image_bytes: Optional[bytes]) -> dict:
    """Send image to MedGemma on Vertex AI and return structured findings dict."""
    if not image_bytes:
        return VLMOutput().model_dump()

    raw = call_medgemma(
        prompt=VLM_STRUCTURED_PROMPT,
        image_bytes=image_bytes,
        max_tokens=MAX_TOKENS_VLM,
    )
    logger.info(f"VLM raw: {raw[:120]}")
    return _parse_vlm_json(raw).model_dump()

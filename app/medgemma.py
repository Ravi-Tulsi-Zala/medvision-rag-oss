"""
Vertex AI Client — MedGemma via Personal GCP Account

AUTH: Uses google.auth.default() which covers all environments:
  - Local:       gcloud auth application-default login  (run once)
  - HF Spaces:   GOOGLE_APPLICATION_CREDENTIALS_JSON secret (full JSON string)
  - Cloud Run:   Attached service account (automatic, no setup)

PERSONAL GCP ACCOUNT BENEFITS:
  - No org policy restrictions
  - Can create service account keys freely
  - Full $300 free credits
  - Full control over billing and IAM
"""

import base64
import json
import logging
import os
import tempfile
from typing import Optional

import google.auth
import google.auth.transport.requests
import requests as http_requests

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID     = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION       = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID", "")
VERTEX_DEDICATED_ENDPOINT_ID = os.getenv("VERTEX_DEDICATED_ENDPOINT_ID", "")

MAX_TOKENS_VLM = 256
MAX_TOKENS_LLM = 1024
TEMPERATURE    = 0.3

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _setup_credentials_from_env():
    """
    If GOOGLE_APPLICATION_CREDENTIALS_JSON is set (HF Spaces secret),
    write it to a temp file and point GOOGLE_APPLICATION_CREDENTIALS at it.
    This runs once at module load time.
    """
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    print(creds_json)
    print(GCP_LOCATION)
    print(GCP_PROJECT_ID)
    if creds_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(json.loads(creds_json), tmp)
            tmp.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
            logger.info("Auth: loaded credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            logger.error(f"Failed to write credentials JSON to temp file: {e}")


# Run at import time
_setup_credentials_from_env()


def _get_access_token() -> str:
    """
    Get a short-lived OAuth2 token via Application Default Credentials.
    Works in all environments after _setup_credentials_from_env().
    """
    try:
        credentials, project = google.auth.default(scopes=SCOPES)
        if not os.getenv("GCP_PROJECT_ID") and project:
            os.environ["GCP_PROJECT_ID"] = project
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except google.auth.exceptions.DefaultCredentialsError:
        raise EnvironmentError(
            "\n❌ GCP credentials not found. Choose one option:\n\n"
            "LOCAL DEV:\n"
            "  gcloud auth application-default login\n"
            "  gcloud config set project YOUR_PROJECT_ID\n\n"
            "HF SPACES / RENDER:\n"
            "  Add secret: GOOGLE_APPLICATION_CREDENTIALS_JSON = <full service account JSON>\n"
            "  (Download JSON key from GCP Console → IAM → Service Accounts → Keys)\n"
        )

def _get_endpoint_url() -> str:
    project  = os.getenv("GCP_PROJECT_ID", GCP_PROJECT_ID)
    location = GCP_LOCATION
    endpoint = VERTEX_ENDPOINT_ID
    dedicated_endpoint = VERTEX_DEDICATED_ENDPOINT_ID
    if not all([project, location, endpoint]):
        raise EnvironmentError(
            "Missing .env vars: GCP_PROJECT_ID, GCP_LOCATION, VERTEX_ENDPOINT_ID"
        )
    return (
        f"https://{dedicated_endpoint}/v1"
        f"/projects/{project}/locations/{location}"
        f"/endpoints/{endpoint}/chat/completions"
    )


def call_medgemma(
    prompt: str,
    image_bytes: Optional[bytes] = None,
    max_tokens: int = MAX_TOKENS_LLM,
) -> str:
    """
    Call MedGemma on Vertex AI endpoint via OpenAI-compatible chat completions.

    Args:
        prompt:      Text instruction for the model.
        image_bytes: Optional raw image bytes (sent as base64 data URL).
        max_tokens:  Max tokens to generate.

    Returns:
        Model response text string.
    """
    content = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "google/medgemma-4b-it",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }
    
    print(f"Payload for Vertex AI: {json.dumps(payload)}")  # Debug log (truncated)

    try:
        token = _get_access_token()
        print(f"Got access token: {token}")
        url   = _get_endpoint_url()

        resp = http_requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Vertex AI OK — {len(text)} chars returned")
        return text

    except EnvironmentError as e:
        logger.error(str(e))
        return str(e)
    except http_requests.exceptions.HTTPError as e:
        status = e.response.status_code
        body   = e.response.text[:400]
        logger.error(f"Vertex AI HTTP {status}: {body}")
        return f"Vertex AI error {status}: {body}"
    except Exception as e:
        logger.error(f"Vertex AI call failed: {e}")
        return f"Vertex AI call failed: {str(e)}"

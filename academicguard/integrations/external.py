"""External tool integration adapters and authentication helpers.

Supported external services:
- Turnitin iThenticate (via TurnitinAdapter in plagiarism.py)
- Copyscape Premium (via CopyscapeAdapter in plagiarism.py)
- Grammarly Business API (enterprise only)
- CrossRef (free, open)
- OpenAI Text Classifier (deprecated; uses our own AI detector instead)
- ZeroGPT API (third-party AI detector)
- GPTZero API (third-party AI detector)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class ExternalServiceStatus:
    name: str
    available: bool
    auth_method: str
    env_vars: list[str]
    note: str = ""


def check_available_services() -> list[ExternalServiceStatus]:
    """Return status of all configured external services."""
    services = [
        ExternalServiceStatus(
            name="Turnitin iThenticate",
            available=bool(os.getenv("TURNITIN_API_KEY")),
            auth_method="Bearer token",
            env_vars=["TURNITIN_API_KEY"],
            note="Requires institutional license",
        ),
        ExternalServiceStatus(
            name="Copyscape Premium",
            available=bool(os.getenv("COPYSCAPE_USER") and os.getenv("COPYSCAPE_KEY")),
            auth_method="Username + API key",
            env_vars=["COPYSCAPE_USER", "COPYSCAPE_KEY"],
            note="Pay-per-search pricing",
        ),
        ExternalServiceStatus(
            name="ZeroGPT AI Detector",
            available=bool(os.getenv("ZEROGPT_API_KEY")),
            auth_method="API key",
            env_vars=["ZEROGPT_API_KEY"],
            note="Free tier available",
        ),
        ExternalServiceStatus(
            name="GPTZero AI Detector",
            available=bool(os.getenv("GPTZERO_API_KEY")),
            auth_method="API key",
            env_vars=["GPTZERO_API_KEY"],
            note="Free tier available",
        ),
        ExternalServiceStatus(
            name="CrossRef Metadata",
            available=True,
            auth_method="None (free open API)",
            env_vars=[],
            note="Always available",
        ),
        ExternalServiceStatus(
            name="LanguageTool (local)",
            available=True,
            auth_method="None",
            env_vars=[],
            note="Downloads on first use (~500MB)",
        ),
        ExternalServiceStatus(
            name="LanguageTool (remote server)",
            available=bool(os.getenv("LANGUAGETOOL_URL")),
            auth_method="URL",
            env_vars=["LANGUAGETOOL_URL"],
            note="Self-hosted or premium API",
        ),
    ]
    return services


# ------------------------------------------------------------------ #
# ZeroGPT adapter
# ------------------------------------------------------------------ #

class ZeroGPTAdapter:
    """ZeroGPT AI text detection API."""

    BASE_URL = "https://api.zerogpt.com/api/detect/detectText"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ZEROGPT_API_KEY", "")

    def detect(self, text: str) -> dict:
        if not self.api_key:
            raise ValueError("ZEROGPT_API_KEY not set. Get a key at https://zerogpt.com/api")
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                self.BASE_URL,
                headers={"ApiKey": self.api_key, "Content-Type": "application/json"},
                json={"input_text": text[:5000]},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "source": "ZeroGPT",
            "ai_probability": data.get("fakePercentage", 0) / 100.0,
            "is_ai": data.get("isHuman", True) is False,
            "ai_words": data.get("aiWords", 0),
            "human_words": data.get("humanWords", 0),
            "sentences": data.get("sentences", []),
        }


# ------------------------------------------------------------------ #
# GPTZero adapter
# ------------------------------------------------------------------ #

class GPTZeroAdapter:
    """GPTZero AI text detection API."""

    BASE_URL = "https://api.gptzero.me/v2/predict/text"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY", "")

    def detect(self, text: str) -> dict:
        if not self.api_key:
            raise ValueError("GPTZERO_API_KEY not set. Get a key at https://gptzero.me/api")
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"document": text[:10000]},
            )
            resp.raise_for_status()
            data = resp.json()
        doc = data.get("documents", [{}])[0]
        return {
            "source": "GPTZero",
            "ai_probability": doc.get("completely_generated_prob", 0.0),
            "human_probability": doc.get("average_generated_prob", 1.0),
            "classification": doc.get("predicted_class", "unknown"),
            "sentences": doc.get("sentences", []),
        }


# ------------------------------------------------------------------ #
# Grammarly adapter (enterprise)
# ------------------------------------------------------------------ #

class GrammarlyAdapter:
    """
    Grammarly Business/Enterprise API adapter.
    Note: Grammarly's API is enterprise-only (not publicly available).
    This adapter uses the unofficial text-check endpoint for research.
    For production, use Grammarly Editor SDK or the official Business API.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GRAMMARLY_API_KEY", "")

    def check(self, text: str) -> dict:
        if not self.api_key:
            raise ValueError(
                "GRAMMARLY_API_KEY not set.\n"
                "Grammarly Business API requires an enterprise account.\n"
                "Alternative: use the free LanguageTool integration (default)."
            )
        # Grammarly does not have a public REST API; placeholder for SDK integration
        return {
            "source": "Grammarly",
            "note": "Grammarly integration requires the Grammarly Editor SDK or Business API.",
            "sdk_url": "https://developer.grammarly.com/",
        }


# ------------------------------------------------------------------ #
# Service environment setup helper
# ------------------------------------------------------------------ #

ENV_SETUP_GUIDE = """
AcademicGuard External Service Configuration
============================================

Set these environment variables to enable external service integrations.
All services are OPTIONAL -- AcademicGuard works fully offline without them.

  # Turnitin iThenticate (institutional license required)
  export TURNITIN_API_KEY="your_turnitin_api_key"

  # Copyscape Premium (pay-per-search)
  export COPYSCAPE_USER="your_username"
  export COPYSCAPE_KEY="your_api_key"

  # ZeroGPT AI detector (free tier available)
  export ZEROGPT_API_KEY="your_zerogpt_key"

  # GPTZero AI detector (free tier available)
  export GPTZERO_API_KEY="your_gptzero_key"

  # LanguageTool self-hosted server (optional, for privacy)
  export LANGUAGETOOL_URL="http://localhost:8010"

Add these to your ~/.bashrc, ~/.zshrc, or a .env file.
Use `academicguard services` to verify which services are active.
"""

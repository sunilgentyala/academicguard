"""
AcademicGuard -- Local service status checker.
All detection is self-contained. This module reports what is available locally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServiceStatus:
    name: str
    available: bool
    note: str


def check_available_services() -> list[ServiceStatus]:
    """Report availability of local detection components."""
    services = []

    # GPT-2 (AI detector)
    try:
        import transformers
        import torch
        services.append(ServiceStatus(
            name="GPT-2 (AI Detection -- GLTR + Perplexity)",
            available=True,
            note=f"transformers {transformers.__version__}, torch {torch.__version__}",
        ))
    except ImportError:
        services.append(ServiceStatus(
            name="GPT-2 (AI Detection -- GLTR + Perplexity)",
            available=False,
            note="Install: pip install transformers torch  (heuristic fallback is active)",
        ))

    # Heuristic AI signals (always available)
    services.append(ServiceStatus(
        name="Heuristic AI Signals (Burstiness, Zipf, Yule-K, Hapax, N-gram, Stylometrics)",
        available=True,
        note="Always available -- pure Python, no dependencies",
    ))

    # Winnowing plagiarism
    services.append(ServiceStatus(
        name="Winnowing Fingerprinter (MOSS-style plagiarism)",
        available=True,
        note="Always available -- pure Python implementation",
    ))

    # MinHash
    try:
        import datasketch
        services.append(ServiceStatus(
            name="MinHash / LSH (fast approximate plagiarism)",
            available=True,
            note=f"datasketch {datasketch.__version__}",
        ))
    except ImportError:
        services.append(ServiceStatus(
            name="MinHash / LSH (fast approximate plagiarism)",
            available=False,
            note="Install: pip install datasketch  (set-Jaccard fallback is active)",
        ))

    # TF-IDF
    services.append(ServiceStatus(
        name="TF-IDF Cosine Similarity (paraphrase plagiarism)",
        available=True,
        note="Always available -- pure Python implementation",
    ))

    # CrossRef
    try:
        import httpx
        services.append(ServiceStatus(
            name="CrossRef Metadata Search (free, open)",
            available=True,
            note="Free REST API -- no API key required",
        ))
    except ImportError:
        services.append(ServiceStatus(
            name="CrossRef Metadata Search (free, open)",
            available=False,
            note="Install: pip install httpx",
        ))

    # LanguageTool
    try:
        import language_tool_python
        services.append(ServiceStatus(
            name="LanguageTool (grammar checker)",
            available=True,
            note="Local Java server -- downloads on first use",
        ))
    except ImportError:
        services.append(ServiceStatus(
            name="LanguageTool (grammar checker)",
            available=False,
            note="Install: pip install language-tool-python",
        ))

    # spaCy
    try:
        import spacy
        spacy.load("en_core_web_sm")
        services.append(ServiceStatus(
            name="spaCy en_core_web_sm (sentence analysis)",
            available=True,
            note="Local model loaded",
        ))
    except Exception:
        services.append(ServiceStatus(
            name="spaCy en_core_web_sm (sentence analysis)",
            available=False,
            note="Run: python -m spacy download en_core_web_sm",
        ))

    return services


ENV_SETUP_GUIDE = """
AcademicGuard -- All Detection Is Local (No External APIs)
==========================================================

All analysis runs on your machine. No API keys required.

Optional: improve accuracy by ensuring all dependencies are installed:

  pip install transformers torch        # GPT-2 for GLTR + perplexity AI detection
  pip install datasketch                # MinHash/LSH for fast plagiarism
  pip install language-tool-python      # grammar checker (downloads Java server)
  python -m spacy download en_core_web_sm  # sentence structure analysis

To check what is available on your machine:
  academicguard services
"""

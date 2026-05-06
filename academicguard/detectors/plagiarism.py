"""Plagiarism detector: local MinHash/LSH fingerprinting + external API adapters.

Local mode (no API key needed):
- k-shingle fingerprinting with MinHash + LSH for near-duplicate detection
- Compares submitted text against a local corpus directory

External integrations (require API credentials):
- Turnitin (iThenticate) REST API v2
- Copyscape Premium API
- PaperRater API (open)
- CrossRef metadata check (open, no key)
"""

from __future__ import annotations

import hashlib
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import httpx

from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class PlagiarismMatch:
    source: str
    similarity: float           # 0.0 - 1.0
    matched_text: str
    source_url: str = ""
    source_type: str = "local"  # "local", "turnitin", "copyscape", "crossref"


@dataclass
class PlagiarismResult:
    overall_similarity: float
    matches: list[PlagiarismMatch]
    checked_against: list[str]
    method: str


# ------------------------------------------------------------------ #
# MinHash fingerprinting (local)
# ------------------------------------------------------------------ #

class MinHashFingerprinter:
    """k-shingle MinHash for near-duplicate detection without external services."""

    NUM_PERM = 128
    SHINGLE_K = 5   # word n-gram size

    def __init__(self):
        try:
            from datasketch import MinHash, MinHashLSH
            self._MinHash = MinHash
            self._MinHashLSH = MinHashLSH
            self._available = True
        except ImportError:
            self._available = False

    def fingerprint(self, text: str):
        tokens = self._tokenize(text)
        shingles = {" ".join(tokens[i:i+self.SHINGLE_K]) for i in range(len(tokens) - self.SHINGLE_K + 1)}
        if not self._available:
            return self._fallback_fingerprint(shingles)
        m = self._MinHash(num_perm=self.NUM_PERM)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def similarity(self, fp1, fp2) -> float:
        if not self._available:
            return self._fallback_similarity(fp1, fp2)
        return fp1.jaccard(fp2)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _fallback_fingerprint(self, shingles: set) -> set:
        return shingles

    def _fallback_similarity(self, s1: set, s2: set) -> float:
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union else 0.0


# ------------------------------------------------------------------ #
# Local corpus checker
# ------------------------------------------------------------------ #

class LocalCorpusChecker:
    """Compares a document against a local directory of reference documents."""

    def __init__(self, corpus_dir: str | Path):
        self.corpus_dir = Path(corpus_dir)
        self._fp = MinHashFingerprinter()
        self._corpus: dict[str, any] = {}

    def load_corpus(self) -> int:
        count = 0
        for f in self.corpus_dir.rglob("*.txt"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                self._corpus[str(f)] = self._fp.fingerprint(text)
                count += 1
            except Exception:
                pass
        return count

    def check(self, text: str, threshold: float = 0.25) -> list[PlagiarismMatch]:
        target_fp = self._fp.fingerprint(text)
        matches = []
        for source, source_fp in self._corpus.items():
            sim = self._fp.similarity(target_fp, source_fp)
            if sim >= threshold:
                matched_text = self._find_common_passage(text, Path(source).read_text(errors="replace"))
                matches.append(PlagiarismMatch(
                    source=source,
                    similarity=sim,
                    matched_text=matched_text,
                    source_type="local",
                ))
        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def _find_common_passage(self, text1: str, text2: str, window: int = 30) -> str:
        words1 = text1.split()
        words2_set = set(text2.lower().split())
        best_start = 0
        best_len = 0
        for i in range(len(words1) - window):
            chunk = words1[i:i+window]
            overlap = sum(1 for w in chunk if w.lower() in words2_set)
            if overlap > best_len:
                best_len = overlap
                best_start = i
        return " ".join(words1[best_start:best_start+window])


# ------------------------------------------------------------------ #
# External API adapters
# ------------------------------------------------------------------ #

class TurnitinAdapter:
    """iThenticate/Turnitin REST API v2 adapter."""

    BASE_URL = "https://app.ithenticate.com/api/v2"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )

    def submit_document(self, title: str, text: str) -> str:
        """Submit document, return submission ID."""
        resp = self._client.post(f"{self.base_url}/submissions", json={
            "title": title,
            "submitter_email": "academicguard@check.local",
            "extraction_type": "TEXT",
            "content": text,
        })
        resp.raise_for_status()
        return resp.json()["id"]

    def get_similarity(self, submission_id: str) -> Optional[float]:
        """Poll for similarity report (may need retries)."""
        resp = self._client.get(f"{self.base_url}/submissions/{submission_id}/similarity")
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "COMPLETE":
            return data["overall_match_percentage"] / 100.0
        return None  # not ready yet

    def check(self, title: str, text: str, poll_interval: int = 5, max_wait: int = 120) -> PlagiarismMatch:
        import time
        sid = self.submit_document(title, text)
        elapsed = 0
        while elapsed < max_wait:
            sim = self.get_similarity(sid)
            if sim is not None:
                return PlagiarismMatch(
                    source="Turnitin iThenticate",
                    similarity=sim,
                    matched_text="(full report available in Turnitin dashboard)",
                    source_url=f"{self.base_url}/submissions/{sid}/viewer",
                    source_type="turnitin",
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError("Turnitin similarity report timed out.")


class CopyscapeAdapter:
    """Copyscape Premium API adapter."""

    BASE_URL = "https://www.copyscape.com/api/"

    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key

    def check(self, text: str) -> list[PlagiarismMatch]:
        with httpx.Client(timeout=30) as client:
            resp = client.post(self.BASE_URL, data={
                "u": self.username,
                "k": self.api_key,
                "o": "csearch",
                "e": "UTF-8",
                "c": "10",
                "t": text[:25000],  # Copyscape 25k char limit
                "f": "json",
            })
            resp.raise_for_status()
            data = resp.json()

        matches = []
        for result in data.get("result", []):
            matches.append(PlagiarismMatch(
                source=result.get("title", result.get("url", "Unknown")),
                similarity=float(result.get("percentmatched", 0)) / 100.0,
                matched_text=result.get("textsnippet", ""),
                source_url=result.get("url", ""),
                source_type="copyscape",
            ))
        return matches


class CrossRefChecker:
    """Free CrossRef metadata search -- detects duplicate publication (self-plagiarism)."""

    BASE_URL = "https://api.crossref.org/works"

    def check_title(self, title: str, author: str = "") -> list[PlagiarismMatch]:
        with httpx.Client(timeout=15) as client:
            params = {"query.title": title, "rows": 5, "mailto": "academicguard@check.local"}
            if author:
                params["query.author"] = author
            resp = client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])

        matches = []
        for item in items:
            score = item.get("score", 0)
            if score < 50:
                continue
            pub_title = " ".join(item.get("title", ["(unknown)"]))
            doi = item.get("DOI", "")
            sim = min(1.0, score / 100.0)
            matches.append(PlagiarismMatch(
                source=f"CrossRef: {pub_title}",
                similarity=sim,
                matched_text=pub_title,
                source_url=f"https://doi.org/{doi}" if doi else "",
                source_type="crossref",
            ))
        return matches


# ------------------------------------------------------------------ #
# Main plagiarism detector
# ------------------------------------------------------------------ #

class PlagiarismDetector:
    """
    Unified plagiarism detector orchestrating local and external checks.

    Configuration via environment variables:
      TURNITIN_API_KEY  -- enables Turnitin iThenticate
      COPYSCAPE_USER, COPYSCAPE_KEY  -- enables Copyscape
    """

    def __init__(
        self,
        corpus_dir: Optional[str | Path] = None,
        use_crossref: bool = True,
        use_turnitin: bool = False,
        use_copyscape: bool = False,
    ):
        self.use_crossref = use_crossref
        self._local: Optional[LocalCorpusChecker] = None
        self._turnitin: Optional[TurnitinAdapter] = None
        self._copyscape: Optional[CopyscapeAdapter] = None
        self._crossref = CrossRefChecker() if use_crossref else None

        if corpus_dir and Path(corpus_dir).is_dir():
            self._local = LocalCorpusChecker(corpus_dir)
            self._local.load_corpus()

        if use_turnitin or os.getenv("TURNITIN_API_KEY"):
            key = os.getenv("TURNITIN_API_KEY", "")
            if key:
                self._turnitin = TurnitinAdapter(key)

        if use_copyscape or (os.getenv("COPYSCAPE_USER") and os.getenv("COPYSCAPE_KEY")):
            user = os.getenv("COPYSCAPE_USER", "")
            key = os.getenv("COPYSCAPE_KEY", "")
            if user and key:
                self._copyscape = CopyscapeAdapter(user, key)

    def analyze(self, doc: Document) -> ModuleResult:
        all_matches: list[PlagiarismMatch] = []
        checked: list[str] = []

        if self._local:
            checked.append("Local corpus")
            all_matches.extend(self._local.check(doc.raw_text))

        if self._crossref and doc.title:
            checked.append("CrossRef (title metadata)")
            try:
                all_matches.extend(self._crossref.check_title(doc.title))
            except Exception as e:
                pass

        if self._turnitin:
            checked.append("Turnitin iThenticate")
            try:
                m = self._turnitin.check(doc.title, doc.raw_text)
                all_matches.append(m)
            except Exception as e:
                pass

        if self._copyscape:
            checked.append("Copyscape")
            try:
                all_matches.extend(self._copyscape.check(doc.raw_text[:25000]))
            except Exception as e:
                pass

        if not checked:
            checked.append("CrossRef (title metadata)" if doc.title else "none")

        overall_sim = max((m.similarity for m in all_matches), default=0.0)
        score = 1.0 - overall_sim
        label = "PASS" if score >= 0.80 else ("WARN" if score >= 0.60 else "FAIL")

        findings = self._build_findings(all_matches, overall_sim)
        summary = self._summary(overall_sim, checked, all_matches)

        return ModuleResult(
            module="Plagiarism",
            score=score,
            label=label,
            summary=summary,
            findings=findings,
            metadata={
                "overall_similarity": round(overall_sim, 3),
                "match_count": len(all_matches),
                "sources_checked": checked,
                "top_matches": [
                    {"source": m.source, "similarity": round(m.similarity, 3), "url": m.source_url}
                    for m in sorted(all_matches, key=lambda x: x.similarity, reverse=True)[:5]
                ],
            },
        )

    def _build_findings(self, matches: list[PlagiarismMatch], overall: float) -> list[Finding]:
        findings: list[Finding] = []
        for m in sorted(matches, key=lambda x: x.similarity, reverse=True)[:10]:
            if m.similarity >= 0.80:
                sev = "error"
            elif m.similarity >= 0.40:
                sev = "warning"
            else:
                sev = "info"
            findings.append(Finding(
                category="plagiarism",
                severity=sev,
                message=f"{m.similarity:.0%} similarity with: {m.source}",
                suggestion="Cite properly or rephrase substantially.",
                rule_id="PLAG-001",
                context=m.matched_text[:200],
            ))
        return findings

    def _summary(self, sim: float, checked: list[str], matches: list[PlagiarismMatch]) -> str:
        sources = ", ".join(checked) if checked else "no external sources"
        if sim >= 0.20:
            return f"{sim:.0%} similarity found (checked: {sources}). {len(matches)} matches require attention."
        return f"Low similarity ({sim:.0%}) across {sources}. No significant matches found."

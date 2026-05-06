"""Base style checker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult


class BaseStyleChecker(ABC):
    """Abstract base class for venue-specific style checkers."""

    venue_name: str = "Generic"
    venue_url: str = ""

    def analyze(self, doc: Document) -> ModuleResult:
        findings = self._check(doc)
        error_count = sum(1 for f in findings if f.severity == "error")
        warning_count = sum(1 for f in findings if f.severity == "warning")
        score = max(0.0, 1.0 - (error_count * 0.15 + warning_count * 0.05))
        score = min(1.0, score)
        label = "PASS" if score >= 0.80 else ("WARN" if score >= 0.55 else "FAIL")
        return ModuleResult(
            module="Style",
            score=score,
            label=label,
            summary=self._summary(findings),
            findings=findings,
            metadata={"venue": self.venue_name, "venue_url": self.venue_url},
        )

    @abstractmethod
    def _check(self, doc: Document) -> list[Finding]:
        ...

    def _summary(self, findings: list[Finding]) -> str:
        errors = sum(1 for f in findings if f.severity == "error")
        warnings = sum(1 for f in findings if f.severity == "warning")
        info = sum(1 for f in findings if f.severity == "info")
        if not findings:
            return f"{self.venue_name} style: No issues found."
        return f"{self.venue_name} style: {errors} errors, {warnings} warnings, {info} notes."

    # ------------------------------------------------------------------ #
    # Shared helper checks used by multiple venue checkers
    # ------------------------------------------------------------------ #

    def _check_abstract_length(
        self, doc: Document, min_words: int, max_words: int, rule_id: str
    ) -> Optional[Finding]:
        if not doc.abstract:
            return Finding(
                category="style",
                severity="error",
                message="Abstract is missing or could not be detected.",
                suggestion="Ensure the abstract is clearly labeled.",
                rule_id=rule_id,
            )
        count = len(doc.abstract.split())
        if count < min_words:
            return Finding(
                category="style",
                severity="warning",
                message=f"Abstract is {count} words -- {self.venue_name} requires {min_words}-{max_words} words.",
                suggestion=f"Expand abstract to at least {min_words} words.",
                rule_id=rule_id,
            )
        if count > max_words:
            return Finding(
                category="style",
                severity="error",
                message=f"Abstract is {count} words -- {self.venue_name} limit is {max_words} words.",
                suggestion=f"Trim abstract to {max_words} words or fewer.",
                rule_id=rule_id,
            )
        return None

    def _check_keywords(
        self, doc: Document, min_kw: int, max_kw: int, rule_id: str
    ) -> list[Finding]:
        findings = []
        if not doc.keywords:
            findings.append(Finding(
                category="style",
                severity="error",
                message="No keywords section detected.",
                suggestion="Add a Keywords section following the abstract.",
                rule_id=rule_id,
            ))
        elif len(doc.keywords) < min_kw:
            findings.append(Finding(
                category="style",
                severity="warning",
                message=f"Only {len(doc.keywords)} keywords found; {self.venue_name} recommends {min_kw}-{max_kw}.",
                rule_id=rule_id,
            ))
        elif len(doc.keywords) > max_kw:
            findings.append(Finding(
                category="style",
                severity="warning",
                message=f"{len(doc.keywords)} keywords found; {self.venue_name} limit is {max_kw}.",
                rule_id=rule_id,
            ))
        return findings

    def _check_section_presence(
        self, doc: Document, required: list[str], rule_id: str
    ) -> list[Finding]:
        findings = []
        existing = {s.title.lower() for s in doc.sections}
        for req in required:
            if not any(req.lower() in e for e in existing):
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'Expected section "{req}" not found.',
                    suggestion=f'Add a "{req}" section as required by {self.venue_name}.',
                    rule_id=rule_id,
                ))
        return findings

    def _check_title_length(
        self, doc: Document, max_words: int, rule_id: str
    ) -> Optional[Finding]:
        if not doc.title:
            return Finding(
                category="style",
                severity="warning",
                message="Document title not detected.",
                rule_id=rule_id,
            )
        count = len(doc.title.split())
        if count > max_words:
            return Finding(
                category="style",
                severity="warning",
                message=f"Title has {count} words -- consider keeping it under {max_words} words.",
                suggestion="Shorten title for readability and indexing.",
                rule_id=rule_id,
            )
        return None

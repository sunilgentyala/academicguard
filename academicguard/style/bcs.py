"""BCS style checker.

Covers: BCS (British Computer Society) publications including:
- The Computer Journal (Oxford University Press / BCS)
- BCS conference proceedings (BCS eWiC)
- BCS-themed journals
Based on: The Computer Journal Author Instructions (2024), BCS Style Guide.
"""

from __future__ import annotations

import re

from academicguard.core.document import Document
from academicguard.core.report import Finding
from academicguard.style.base import BaseStyleChecker
from academicguard.style.iet import _AMERICAN_TO_BRITISH  # BCS also uses British English


class BCSStyleChecker(BaseStyleChecker):
    venue_name = "BCS / The Computer Journal"
    venue_url = "https://academic.oup.com/comjnl/pages/General_Instructions"

    def _check(self, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        # Abstract: 100-250 words (The Computer Journal: up to 200)
        f = self._check_abstract_length(doc, 100, 250, "BCS-ABS-001")
        if f:
            findings.append(f)

        # Abstract must be informative (not just a description of intent)
        findings.extend(self._check_abstract_quality(doc))

        # Keywords: 4-6
        findings.extend(self._check_keywords(doc, 4, 8, "BCS-KW-001"))

        # British English
        findings.extend(self._check_british_spelling(doc))

        # Title: no abbreviations, no question marks
        findings.extend(self._check_title(doc))

        # Sections
        findings.extend(self._check_sections(doc))

        # Reference style: BCS/Oxford uses numbered references (Vancouver style)
        findings.extend(self._check_references(doc))

        # Figures and tables
        findings.extend(self._check_figures_tables(doc))

        # Ethical and legal compliance
        findings.extend(self._check_compliance(doc))

        # Manuscript length guideline
        findings.extend(self._check_length(doc))

        return findings

    def _check_abstract_quality(self, doc: Document) -> list[Finding]:
        if not doc.abstract:
            return []
        findings = []
        abstract = doc.abstract.lower()

        # Avoid "This paper ..." as first words (BCS prefers result-forward abstracts)
        if abstract.startswith("this paper") or abstract.startswith("in this paper"):
            findings.append(Finding(
                category="style",
                severity="info",
                message='Abstract begins with "This paper..." -- BCS/Oxford prefers result-forward writing.',
                suggestion='Start with the finding or contribution: "We present...", "A novel method is proposed..."',
                rule_id="BCS-ABS-002",
            ))

        # Abstract should not reference tables or figures
        if re.search(r"\b(fig|figure|table|appendix)\b", abstract, re.I):
            findings.append(Finding(
                category="style",
                severity="error",
                message="Abstract references figures/tables. BCS abstracts must be self-contained.",
                suggestion="Remove all figure/table references from the abstract.",
                rule_id="BCS-ABS-003",
            ))

        return findings

    def _check_british_spelling(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        count = 0
        for american, british in _AMERICAN_TO_BRITISH.items():
            if american == "program":
                continue  # CS context: "program" acceptable
            pattern = re.compile(rf"\b{re.escape(american)}\b", re.I)
            for m in pattern.finditer(text):
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'American spelling "{m.group()}". BCS/Oxford uses British English.',
                    location=f"offset {m.start()}",
                    suggestion=f'Use "{british}".',
                    rule_id="BCS-SPELL-001",
                    context=text[max(0, m.start()-20):m.end()+20],
                ))
                count += 1
                if count >= 15:
                    return findings
        return findings

    def _check_title(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.title:
            return findings

        # No abbreviations in titles (BCS guideline)
        abbr_pattern = re.compile(r"\b[A-Z]{2,6}\b")
        abbrs = abbr_pattern.findall(doc.title)
        # Allow a few common abbreviations
        common_ok = {"AI", "ML", "DL", "IoT", "API", "HTTP", "TCP", "IP", "5G", "LTE"}
        unexplained = [a for a in abbrs if a not in common_ok]
        if unexplained:
            findings.append(Finding(
                category="style",
                severity="warning",
                message=f'Unexplained acronyms in title: {unexplained}.',
                suggestion="BCS guidelines: spell out acronyms in the title unless universally known.",
                rule_id="BCS-TTL-001",
                context=doc.title,
            ))

        # No question marks in title
        if "?" in doc.title:
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Question mark in title. BCS/Oxford discourages question-form titles.",
                suggestion="Rephrase as a declarative statement.",
                rule_id="BCS-TTL-002",
            ))

        # Max 15 words
        f = self._check_title_length(doc, 15, "BCS-TTL-003")
        if f:
            findings.append(f)

        return findings

    def _check_sections(self, doc: Document) -> list[Finding]:
        required = ["Introduction", "Conclusion"]
        findings = self._check_section_presence(doc, required, "BCS-SEC-001")

        # BCS: no subsection deeper than 3 levels
        text = doc.raw_text
        deep_sections = re.findall(r"^\d+\.\d+\.\d+\.\d+\s+", text, re.M)
        if deep_sections:
            findings.append(Finding(
                category="style",
                severity="warning",
                message=f"Section nesting exceeds 3 levels ({len(deep_sections)} instances).",
                suggestion="BCS recommends no more than 3 levels of section hierarchy.",
                rule_id="BCS-SEC-002",
            ))
        return findings

    def _check_references(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.references:
            return [Finding(
                category="style",
                severity="warning",
                message="No references section detected.",
                rule_id="BCS-REF-001",
            )]

        # BCS/Oxford uses numbered references in square brackets [1]
        # Check for author-year style
        author_year = re.compile(r"\([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?,\s*\d{4}\)")
        if author_year.search(doc.raw_text):
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Author-year citation style detected. BCS/The Computer Journal uses numbered [N] references.",
                suggestion='Replace "(Author, Year)" with "[N]" numeric citations.',
                rule_id="BCS-REF-002",
            ))

        # Check journal name abbreviation
        for i, ref in enumerate(doc.references[:5]):
            # BCS uses full journal names, not abbreviations
            if re.search(r"\bComput\.\b|\bInt\.\b|\bJ\.\b|\bTrans\.\b", ref):
                findings.append(Finding(
                    category="style",
                    severity="info",
                    message=f"Reference {i+1} uses abbreviated journal name. BCS prefers full names.",
                    suggestion="Spell out journal names in full.",
                    rule_id="BCS-REF-003",
                    context=ref[:100],
                ))
                break
        return findings

    def _check_figures_tables(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text

        # BCS: Figures below the relevant text, Tables above
        # Check figure/table numbering consistency
        fig_nums = re.findall(r"(?i)fig(?:ure)?\.?\s*(\d+)", text)
        tbl_nums = re.findall(r"(?i)table\s+(\d+)", text)

        if fig_nums:
            nums = [int(n) for n in fig_nums]
            expected = list(range(1, max(nums) + 1))
            missing = [n for n in expected if n not in nums]
            if missing:
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f"Figure numbering may be non-sequential (missing: {missing}).",
                    suggestion="Ensure figures are numbered consecutively.",
                    rule_id="BCS-FIG-001",
                ))
        return findings

    def _check_compliance(self, doc: Document) -> list[Finding]:
        findings = []
        text_lower = doc.raw_text.lower()

        has_ack = "acknowledgement" in text_lower or "acknowledgment" in text_lower
        if not has_ack:
            findings.append(Finding(
                category="style",
                severity="warning",
                message="No Acknowledgements section. BCS requires funding and support disclosure.",
                suggestion="Add Acknowledgements (British spelling) including funding sources and institutional support.",
                rule_id="BCS-ACK-001",
            ))

        has_coi = any(k in text_lower for k in [
            "conflict of interest", "competing interest", "the authors declare"
        ])
        if not has_coi:
            findings.append(Finding(
                category="style",
                severity="error",
                message="No conflict of interest declaration found.",
                suggestion='Add: "Conflict of interest statement: None declared." or disclose interests.',
                rule_id="BCS-COI-001",
            ))
        return findings

    def _check_length(self, doc: Document) -> list[Finding]:
        word_count = doc.word_count
        # The Computer Journal: typically 4000-8000 words for a full article
        if word_count > 10000:
            return [Finding(
                category="style",
                severity="info",
                message=f"Manuscript is {word_count} words. BCS/The Computer Journal recommends 4000-8000 words.",
                suggestion="Consider condensing or splitting into multiple papers.",
                rule_id="BCS-LEN-001",
            )]
        if word_count < 2000:
            return [Finding(
                category="style",
                severity="warning",
                message=f"Manuscript is only {word_count} words. May be too short for a full article.",
                suggestion="Target 4000+ words or consider a correspondence/letter format.",
                rule_id="BCS-LEN-002",
            )]
        return []

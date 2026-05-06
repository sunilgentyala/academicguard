"""Elsevier style checker.

Covers: ScienceDirect journals (e.g., Expert Systems with Applications,
Computers & Security, Information Sciences, Pattern Recognition, etc.)
Based on: Elsevier Author Guidelines, Guide for Authors (2024).
"""

from __future__ import annotations

import re

from academicguard.core.document import Document
from academicguard.core.report import Finding
from academicguard.style.base import BaseStyleChecker


class ElsevierStyleChecker(BaseStyleChecker):
    venue_name = "Elsevier"
    venue_url = "https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions"

    def _check(self, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        # Abstract: 150-300 words (journal-dependent; 250 typical)
        f = self._check_abstract_length(doc, 150, 300, "ELS-ABS-001")
        if f:
            findings.append(f)

        # Structured abstract check (some Elsevier journals require it)
        findings.extend(self._check_structured_abstract(doc))

        # Keywords: 4-8 for most Elsevier journals
        findings.extend(self._check_keywords(doc, 4, 8, "ELS-KW-001"))
        findings.extend(self._check_keyword_format(doc))

        # Highlights (mandatory for most Elsevier journals)
        findings.extend(self._check_highlights(doc))

        # CRediT author contribution statement
        findings.extend(self._check_credit_statement(doc))

        # References: Elsevier uses numbered references [1] or Name (year)
        findings.extend(self._check_reference_format(doc))

        # Section structure
        findings.extend(self._check_sections(doc))

        # Ethical compliance statements
        findings.extend(self._check_ethical_statements(doc))

        # Figure captions and table notes
        findings.extend(self._check_figure_table_captions(doc))

        # Declaration of competing interests
        findings.extend(self._check_competing_interests(doc))

        return findings

    def _check_structured_abstract(self, doc: Document) -> list[Finding]:
        if not doc.abstract:
            return []
        structured_headers = ["background", "objective", "method", "result", "conclusion"]
        has_structure = any(h in doc.abstract.lower() for h in structured_headers)
        word_count = len(doc.abstract.split())
        # Only flag if abstract is long enough to warrant structure
        if not has_structure and word_count > 180:
            return [Finding(
                category="style",
                severity="info",
                message="Consider using a structured abstract with Background, Objective, Methods, Results, and Conclusions.",
                suggestion="Many Elsevier journals (e.g., Computers & Security) require structured abstracts.",
                rule_id="ELS-ABS-002",
            )]
        return []

    def _check_keyword_format(self, doc: Document) -> list[Finding]:
        findings = []
        for kw in doc.keywords:
            # Elsevier: no punctuation at end, title case or lowercase
            if kw.endswith((".", ",", ";")):
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'Keyword "{kw}" should not end with punctuation.',
                    suggestion="Remove trailing punctuation from keywords.",
                    rule_id="ELS-KW-002",
                    context=kw,
                ))
        return findings

    def _check_highlights(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        if "highlights" not in text_lower and "highlight" not in text_lower:
            return [Finding(
                category="style",
                severity="error",
                message="Elsevier journals require a Highlights section (3-5 bullet points, max 85 characters each).",
                suggestion='Add a "Highlights" section before or after the abstract with 3-5 bullet points summarizing key findings.',
                rule_id="ELS-HLT-001",
            )]

        # Check highlight length
        highlight_section = re.search(
            r"highlights?[:\s]*(.*?)(?=\n\n|\babstract\b|\bkeywords?\b)", doc.raw_text, re.I | re.S
        )
        if highlight_section:
            bullets = re.findall(r"[*\-•]\s*(.+)", highlight_section.group(1))
            for b in bullets:
                if len(b) > 85:
                    return [Finding(
                        category="style",
                        severity="warning",
                        message=f'Highlight exceeds 85-character Elsevier limit: "{b[:60]}..."',
                        suggestion="Shorten each highlight to 85 characters maximum.",
                        rule_id="ELS-HLT-002",
                    )]
        return []

    def _check_credit_statement(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_credit = "credit" in text_lower or "author contribution" in text_lower
        if not has_credit:
            return [Finding(
                category="style",
                severity="warning",
                message="Elsevier requires a CRediT Author Contribution Statement.",
                suggestion='Add: "Author Contributions: [Name]: Conceptualization, Methodology... [Name]: Writing -- Review & Editing..."',
                rule_id="ELS-CREDIT-001",
            )]
        return []

    def _check_reference_format(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.references:
            return [Finding(
                category="style",
                severity="warning",
                message="No references section detected.",
                rule_id="ELS-REF-001",
            )]

        for i, ref in enumerate(doc.references[:10]):
            # Elsevier Harvard style: Author, A., Author, B., Year. Title. Journal Volume, Pages.
            # Elsevier numbered: [1] Author, A. (Year). Title. Journal, Volume(Issue), Pages.
            # Check DOI presence
            if not re.search(r"doi|https?://dx\.doi|10\.\d{4}", ref, re.I):
                if i < 3:  # flag only first few for brevity
                    findings.append(Finding(
                        category="style",
                        severity="info",
                        message=f"Reference {i+1} may be missing a DOI.",
                        suggestion="Elsevier strongly recommends including DOIs for all references.",
                        rule_id="ELS-REF-002",
                        context=ref[:100],
                    ))
        return findings

    def _check_sections(self, doc: Document) -> list[Finding]:
        return self._check_section_presence(
            doc,
            ["Introduction", "Methods", "Results", "Discussion", "Conclusion"],
            "ELS-SEC-001",
        )

    def _check_ethical_statements(self, doc: Document) -> list[Finding]:
        findings = []
        text_lower = doc.raw_text.lower()
        has_ethics = any(k in text_lower for k in [
            "ethics", "ethical approval", "irb", "institutional review",
            "informed consent", "data availability", "funding"
        ])
        if not has_ethics:
            findings.append(Finding(
                category="style",
                severity="warning",
                message="No ethical/compliance statements detected.",
                suggestion="Add Funding, Data Availability, and Ethics/IRB statements as required by Elsevier.",
                rule_id="ELS-ETH-001",
            ))
        return findings

    def _check_figure_table_captions(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        # Figures should be self-explanatory
        fig_captions = re.findall(r"(?i)fig(?:ure)?\.?\s*\d+[.:]\s*(.{20,200})", text)
        for cap in fig_captions:
            if len(cap.split()) < 5:
                findings.append(Finding(
                    category="style",
                    severity="info",
                    message=f'Figure caption seems too short: "{cap[:60]}"',
                    suggestion="Elsevier captions should be self-explanatory without reading the full text.",
                    rule_id="ELS-FIG-001",
                ))
                break
        return findings

    def _check_competing_interests(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_coi = any(k in text_lower for k in [
            "competing interest", "conflict of interest", "declaration of interest",
            "the authors declare"
        ])
        if not has_coi:
            return [Finding(
                category="style",
                severity="error",
                message="Elsevier requires a Declaration of Competing Interests.",
                suggestion='Add: "Declaration of Competing Interests: The authors declare no competing interests." or disclose relevant interests.',
                rule_id="ELS-COI-001",
            )]
        return []

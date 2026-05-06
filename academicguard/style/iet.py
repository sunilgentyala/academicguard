"""IET style checker.

Covers: IET (Institution of Engineering and Technology) journals and conference proceedings.
Includes: IET Communications, IET Networks, IET Cyber-Systems and Security, etc.
Based on: IET Author Guide (2024), IET style guide.
Note: IET publications use British English conventions.
"""

from __future__ import annotations

import re

from academicguard.core.document import Document
from academicguard.core.report import Finding
from academicguard.style.base import BaseStyleChecker


# British vs. American spelling pairs (IET uses British)
_AMERICAN_TO_BRITISH: dict[str, str] = {
    "analyze": "analyse",
    "analyzes": "analyses",
    "analyzed": "analysed",
    "analyzing": "analysing",
    "optimize": "optimise",
    "optimizes": "optimises",
    "optimized": "optimised",
    "optimizing": "optimising",
    "recognize": "recognise",
    "recognizes": "recognises",
    "recognized": "recognised",
    "recognizing": "recognising",
    "organize": "organise",
    "organized": "organised",
    "organizing": "organising",
    "utilization": "utilisation",
    "utilizations": "utilisations",
    "realization": "realisation",
    "initialization": "initialisation",
    "characterize": "characterise",
    "characterization": "characterisation",
    "synchronize": "synchronise",
    "synchronization": "synchronisation",
    "color": "colour",
    "colors": "colours",
    "colored": "coloured",
    "neighbor": "neighbour",
    "neighbors": "neighbours",
    "behaviour": "behaviour",  # already British (keep)
    "center": "centre",
    "centers": "centres",
    "fiber": "fibre",
    "meter": "metre",
    "meters": "metres",
    "defense": "defence",
    "license": "licence",
    "practice": "practise",  # as verb
    "catalog": "catalogue",
    "dialog": "dialogue",
    "program": "programme",   # in British English (except computer program)
}


class IETStyleChecker(BaseStyleChecker):
    venue_name = "IET"
    venue_url = "https://ietresearch.onlinelibrary.wiley.com/hub/authors"

    def _check(self, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        # Abstract: 100-200 words (IET typically shorter than IEEE)
        f = self._check_abstract_length(doc, 100, 200, "IET-ABS-001")
        if f:
            findings.append(f)

        # Abstract must not contain references or mathematical expressions
        findings.extend(self._check_abstract_content(doc))

        # Keywords: 4-8
        findings.extend(self._check_keywords(doc, 4, 8, "IET-KW-001"))

        # British English spelling
        findings.extend(self._check_british_spelling(doc))

        # References: IET uses numbered [1] or [1,2,3] style
        findings.extend(self._check_references(doc))

        # Sections
        findings.extend(self._check_sections(doc))

        # IET units and notation
        findings.extend(self._check_notation(doc))

        # Author biography (IET journals require brief bio)
        findings.extend(self._check_author_bio(doc))

        # Funding acknowledgment
        findings.extend(self._check_funding(doc))

        # IET Data availability statement
        findings.extend(self._check_data_availability(doc))

        return findings

    def _check_abstract_content(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.abstract:
            return findings
        # No citations in abstract
        if re.search(r"\[\d+\]|\([A-Z][a-z]+.*?\d{4}\)", doc.abstract):
            findings.append(Finding(
                category="style",
                severity="error",
                message="Abstract contains references/citations. IET abstracts must be self-contained.",
                suggestion="Remove all citations from the abstract.",
                rule_id="IET-ABS-002",
            ))
        # No equations in abstract
        if re.search(r"=|\\frac|\\sum|\\prod", doc.abstract):
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Abstract appears to contain mathematical expressions.",
                suggestion="IET abstracts should avoid equations; describe results verbally.",
                rule_id="IET-ABS-003",
            ))
        return findings

    def _check_british_spelling(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        count = 0
        for american, british in _AMERICAN_TO_BRITISH.items():
            # Skip "program" for computer science context
            if american == "program" and re.search(r"computer program|program code|program\b", text, re.I):
                continue
            pattern = re.compile(rf"\b{re.escape(american)}\b", re.I)
            for m in pattern.finditer(text):
                # Check context -- avoid flagging already-correct British spellings
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'American spelling "{m.group()}". IET uses British English.',
                    location=f"offset {m.start()}",
                    suggestion=f'Use "{british}" (British spelling).',
                    rule_id="IET-SPELL-001",
                    context=text[max(0, m.start()-20):m.end()+20],
                ))
                count += 1
                if count >= 15:  # cap findings
                    return findings
        return findings

    def _check_references(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.references:
            return [Finding(
                category="style",
                severity="warning",
                message="No references detected.",
                rule_id="IET-REF-001",
            )]

        for i, ref in enumerate(doc.references[:5]):
            # IET format: [N] Surname, I.: 'Title', Journal, Year, Vol(Issue), pp. Pages
            # Check for journal abbreviation
            if not re.search(r"vol\.|volume|issue|pp\.|pages?", ref, re.I):
                findings.append(Finding(
                    category="style",
                    severity="info",
                    message=f"Reference {i+1} may be missing volume/issue/page details.",
                    suggestion="IET format: Author(s): 'Title'. Journal, Year, Vol(Issue), pp. xx-yy.",
                    rule_id="IET-REF-002",
                    context=ref[:120],
                ))
        return findings

    def _check_sections(self, doc: Document) -> list[Finding]:
        return self._check_section_presence(
            doc,
            ["Introduction", "Conclusion"],
            "IET-SEC-001",
        )

    def _check_notation(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text

        # IET: use "equation (N)" not "Eq. (N)"
        eq_abbrev = re.findall(r"\bEq\.\s*\(\d+\)", text)
        if eq_abbrev:
            findings.append(Finding(
                category="style",
                severity="info",
                message=f'"Eq. (N)" detected. IET style uses "equation (N)" in full.',
                suggestion='Replace "Eq. (N)" with "equation (N)".',
                rule_id="IET-NOT-001",
            ))

        # IET: "Figure" (not "Fig.") in body text for some IET publications
        fig_abbrev = re.findall(r"\bFig\.\s*\d+\b", text)
        if len(fig_abbrev) > 3:
            findings.append(Finding(
                category="style",
                severity="info",
                message='"Fig." detected. Some IET journals prefer "Figure N" in full.',
                suggestion="Check specific IET journal guide -- some use abbreviation, others do not.",
                rule_id="IET-NOT-002",
            ))
        return findings

    def _check_author_bio(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_bio = any(k in text_lower for k in [
            "received the", "is currently", "is a member", "joined", "biography", "biographies"
        ])
        if not has_bio and len(doc.raw_text.split()) > 2000:
            return [Finding(
                category="style",
                severity="info",
                message="IET journal papers typically include a short author biography.",
                suggestion="Add a 50-100 word biography for each author at the end of the paper.",
                rule_id="IET-BIO-001",
            )]
        return []

    def _check_funding(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_funding = any(k in text_lower for k in [
            "acknowledgment", "acknowledgement", "funding", "grant", "supported by",
            "this work was", "this research was"
        ])
        if not has_funding:
            return [Finding(
                category="style",
                severity="warning",
                message="No acknowledgment/funding section detected.",
                suggestion="IET requires disclosure of funding sources. Add an Acknowledgments section.",
                rule_id="IET-FUND-001",
            )]
        return []

    def _check_data_availability(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_data = any(k in text_lower for k in [
            "data availability", "data sharing", "data access", "dataset", "repository",
            "code availability", "supplementary"
        ])
        if not has_data:
            return [Finding(
                category="style",
                severity="info",
                message="IET encourages a Data Availability Statement.",
                suggestion='Add: "Data Availability Statement: The data that support the findings of this study are available at [URL/DOI]."',
                rule_id="IET-DATA-001",
            )]
        return []

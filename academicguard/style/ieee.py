"""IEEE style checker.

Covers IEEE Transactions, Access, Letters, and Conference (ICASSP, INFOCOM, etc.)
Based on: IEEE Author Center guidelines, IEEE Editorial Style Manual (2023).
"""

from __future__ import annotations

import re
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import Finding
from academicguard.style.base import BaseStyleChecker


class IEEEStyleChecker(BaseStyleChecker):
    """IEEE Transactions / Conference style checker."""

    venue_name = "IEEE"
    venue_url = "https://ieeeauthorcenter.ieee.org"

    # IEEE citation pattern: [1], [2,3], [1]-[4]
    _CITE_PATTERN = re.compile(r"\[(\d+(?:[,\-]\d+)*)\]")
    _NON_IEEE_CITE = re.compile(r"\([\w].*?\d{4}\)")   # author-year style

    # IEEE units: number space unit, SI preferred
    _UNIT_NO_SPACE = re.compile(r"(\d)(GHz|MHz|kHz|Hz|Gbps|Mbps|kbps|bps|ms|us|ns|W|mW|dB|dBm)")

    # et al. formatting
    _ET_AL = re.compile(r"et\. al\.|et al(?!\.)|\betal\b", re.I)

    def _check(self, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        # Abstract
        f = self._check_abstract_length(doc, 150, 250, "IEEE-ABS-001")
        if f:
            findings.append(f)

        # Keywords (IEEE: 3-10 Index Terms)
        findings.extend(self._check_keywords(doc, 3, 10, "IEEE-KW-001"))
        findings.extend(self._check_keyword_capitalization(doc))

        # Title
        f = self._check_title_length(doc, 15, "IEEE-TTL-001")
        if f:
            findings.append(f)
        findings.extend(self._check_title_case(doc))

        # Sections
        findings.extend(self._check_ieee_sections(doc))

        # Citations
        findings.extend(self._check_citation_style(doc))

        # Units
        findings.extend(self._check_units(doc))

        # et al. formatting
        findings.extend(self._check_et_al(doc))

        # Figure/table references
        findings.extend(self._check_figure_references(doc))

        # Acronym definition
        findings.extend(self._check_acronym_definition(doc))

        # Roman numeral sections (IEEE conference style)
        findings.extend(self._check_roman_section_numbers(doc))

        return findings

    def _check_keyword_capitalization(self, doc: Document) -> list[Finding]:
        findings = []
        for kw in doc.keywords:
            # IEEE: keywords lowercase unless proper noun
            if kw != kw.lower() and kw[0].isupper() and not self._is_proper_noun(kw):
                findings.append(Finding(
                    category="style",
                    severity="info",
                    message=f'IEEE Index Term "{kw}" should be lowercase (unless a proper noun).',
                    suggestion=f'Use "{kw.lower()}" unless it is a proper noun or acronym.',
                    rule_id="IEEE-KW-002",
                    context=kw,
                ))
        return findings

    def _is_proper_noun(self, word: str) -> bool:
        acronym_pattern = re.compile(r"^[A-Z]{2,}$")
        return acronym_pattern.match(word) is not None

    def _check_title_case(self, doc: Document) -> list[Finding]:
        if not doc.title:
            return []
        words = doc.title.split()
        minor_words = {"a", "an", "the", "and", "but", "or", "for", "nor",
                       "on", "at", "to", "by", "in", "of", "up", "as", "is"}
        issues = []
        for i, word in enumerate(words[1:], 1):  # skip first word
            clean = re.sub(r"[^\w]", "", word).lower()
            if clean in minor_words and word[0].isupper():
                issues.append(word)
        if issues:
            return [Finding(
                category="style",
                severity="info",
                message=f"IEEE title case: minor words {issues} may not need capitalization.",
                suggestion="Check IEEE title case: capitalize all major words; lowercase articles/prepositions.",
                rule_id="IEEE-TTL-002",
                context=doc.title,
            )]
        return []

    def _check_ieee_sections(self, doc: Document) -> list[Finding]:
        findings = []
        expected = ["introduction", "conclusion"]
        for req in expected:
            if not any(req in s.title.lower() for s in doc.sections):
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'IEEE papers should include a "{req.title()}" section.',
                    suggestion=f"Add a dedicated {req.title()} section.",
                    rule_id="IEEE-SEC-001",
                ))

        # Check for Related Work or Background
        has_related = any(
            any(k in s.title.lower() for k in ["related", "background", "literature", "prior"])
            for s in doc.sections
        )
        if not has_related and len(doc.sections) > 3:
            findings.append(Finding(
                category="style",
                severity="info",
                message='Consider including a "Related Work" or "Background" section.',
                suggestion="IEEE papers typically survey related literature separately.",
                rule_id="IEEE-SEC-002",
            ))
        return findings

    def _check_citation_style(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text

        # Detect author-year citations (not IEEE style)
        author_year = _NON_IEEE_CITE = re.compile(r"\b[A-Z][a-z]+ et al\., \d{4}\b")
        matches = author_year.findall(text)
        if matches:
            findings.append(Finding(
                category="style",
                severity="error",
                message=f"Author-year citations detected ({matches[0]} ...). IEEE uses numeric [N] style.",
                suggestion='Replace "(Author et al., 2023)" with "[N]" numeric references.',
                rule_id="IEEE-CITE-001",
            ))

        # Check citation order (should be sequential first use)
        cite_nums = [int(n) for m in self._CITE_PATTERN.finditer(text)
                     for n in re.split(r"[,\-]", m.group(1)) if n.isdigit()]
        if cite_nums:
            first_appearance: dict[int, int] = {}
            for i, n in enumerate(cite_nums):
                if n not in first_appearance:
                    first_appearance[n] = i
            # Check if numbers appear in non-ascending order at first use
            fa_list = sorted(first_appearance.items(), key=lambda x: x[1])
            nums_in_order = [x[0] for x in fa_list]
            is_sorted = all(nums_in_order[i] <= nums_in_order[i+1] for i in range(len(nums_in_order)-1))
            if not is_sorted and len(nums_in_order) > 2:
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message="IEEE citations should be numbered in order of first appearance.",
                    suggestion="Renumber references so [1] appears before [2], etc.",
                    rule_id="IEEE-CITE-002",
                ))

        return findings

    def _check_units(self, doc: Document) -> list[Finding]:
        findings = []
        for m in self._UNIT_NO_SPACE.finditer(doc.raw_text):
            findings.append(Finding(
                category="style",
                severity="warning",
                message=f'Missing space between number and unit: "{m.group()}"',
                location=f"offset {m.start()}",
                suggestion=f'Use "{m.group(1)} {m.group(2)}" (SI convention).',
                rule_id="IEEE-UNIT-001",
                context=m.group(),
            ))
            if len(findings) >= 5:
                break
        return findings

    def _check_et_al(self, doc: Document) -> list[Finding]:
        findings = []
        for m in self._ET_AL.finditer(doc.raw_text):
            if m.group() != "et al.":
                findings.append(Finding(
                    category="style",
                    severity="warning",
                    message=f'Incorrect "et al." formatting: "{m.group()}"',
                    location=f"offset {m.start()}",
                    suggestion='Use "et al." (italic in IEEE style, period after "al").',
                    rule_id="IEEE-ETAL-001",
                    context=m.group(),
                ))
        return findings

    def _check_figure_references(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        # IEEE: "Fig." not "Figure" in body text; "Table" is spelled out
        fig_full = re.findall(r"\bFigure\s+\d+\b", text)
        if fig_full:
            findings.append(Finding(
                category="style",
                severity="info",
                message=f'"Figure N" should be abbreviated "Fig. N" in IEEE body text ({len(fig_full)} instances).',
                suggestion='Use "Fig." in running text; "Figure" only at start of sentence.',
                rule_id="IEEE-FIG-001",
            ))
        # Check fig/table captions exist
        has_fig = bool(re.search(r"(Fig\.|Figure)\s*\d+", text, re.I))
        has_table = bool(re.search(r"Table\s+[IVX\d]+", text, re.I))
        if not has_fig:
            findings.append(Finding(
                category="style",
                severity="info",
                message="No figure references detected. IEEE papers typically include figures.",
                rule_id="IEEE-FIG-002",
            ))
        return findings

    def _check_acronym_definition(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        # Find parenthetical acronym definitions: Machine Learning (ML)
        defined = set(re.findall(r"\(([A-Z]{2,6})\)", text))
        # Check that each acronym defined in parentheses appears before without definition
        # Simple check: ensure it appears somewhere after its definition
        for acr in defined:
            pattern = re.compile(rf"\b{re.escape(acr)}\b")
            occurrences = list(pattern.finditer(text))
            if len(occurrences) == 1:
                findings.append(Finding(
                    category="style",
                    severity="info",
                    message=f'Acronym "{acr}" is defined but used only once -- consider spelling out.',
                    suggestion="IEEE style: define acronym on first use; omit if used only once.",
                    rule_id="IEEE-ACR-001",
                    context=acr,
                ))
        return findings

    def _check_roman_section_numbers(self, doc: Document) -> list[Finding]:
        findings = []
        has_roman = any(
            re.match(r"^(I{1,3}V?|V?I{0,3}|IX|X{1,3})\.", s.title)
            for s in doc.sections
        )
        has_arabic = any(
            re.match(r"^\d+\.", s.title)
            for s in doc.sections
        )
        if has_arabic and has_roman:
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Mixed Roman and Arabic section numbering detected.",
                suggestion="IEEE Transactions use Roman numerals (I., II., III.); conferences may use Arabic.",
                rule_id="IEEE-SEC-003",
            ))
        return findings

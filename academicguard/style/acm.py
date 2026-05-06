"""ACM style checker.

Covers ACM Digital Library publications: SIGCOMM, CCS, SOSP, CHI, USENIX-affiliated, etc.
Based on: ACM Publication System (TAPS) guidelines, ACM Master Article Template (2024).
"""

from __future__ import annotations

import re

from academicguard.core.document import Document
from academicguard.core.report import Finding
from academicguard.style.base import BaseStyleChecker


class ACMStyleChecker(BaseStyleChecker):
    venue_name = "ACM"
    venue_url = "https://authors.acm.org"

    # ACM CCS taxonomy concepts
    _CCS_PATTERN = re.compile(
        r"CCS\s+concepts?|ACM\s+subject\s+descriptors?|"
        r"Computing\s+Classification\s+System",
        re.I,
    )

    def _check(self, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        # Abstract: 150-250 words
        f = self._check_abstract_length(doc, 100, 250, "ACM-ABS-001")
        if f:
            findings.append(f)

        # Keywords: ACM recommends from ACM CCS taxonomy
        findings.extend(self._check_keywords(doc, 3, 10, "ACM-KW-001"))

        # CCS Concepts
        findings.extend(self._check_ccs_concepts(doc))

        # Title
        f = self._check_title_length(doc, 12, "ACM-TTL-001")
        if f:
            findings.append(f)

        # ACM-specific sections
        findings.extend(self._check_sections(doc))

        # Reference format: ACM uses author-number hybrid or author-year
        findings.extend(self._check_references(doc))

        # ORCID / author information
        findings.extend(self._check_author_info(doc))

        # Artifact availability (ACM reproducibility)
        findings.extend(self._check_artifact_availability(doc))

        # ACM rights / license statement
        findings.extend(self._check_rights(doc))

        # Ethical review statement (ACM CHI, FAccT, etc.)
        findings.extend(self._check_ethics(doc))

        return findings

    def _check_ccs_concepts(self, doc: Document) -> list[Finding]:
        if not self._CCS_PATTERN.search(doc.raw_text):
            return [Finding(
                category="style",
                severity="error",
                message="ACM CCS (Computing Classification System) concepts not found.",
                suggestion='Add "CCS Concepts:" section using ACM CCS taxonomy (https://dl.acm.org/ccs). '
                           'Example: "Security and privacy -> Access control."',
                rule_id="ACM-CCS-001",
            )]
        return []

    def _check_sections(self, doc: Document) -> list[Finding]:
        return self._check_section_presence(
            doc,
            ["Introduction", "Related Work", "Conclusion"],
            "ACM-SEC-001",
        )

    def _check_references(self, doc: Document) -> list[Finding]:
        findings = []
        if not doc.references:
            return [Finding(
                category="style",
                severity="warning",
                message="No references section detected.",
                rule_id="ACM-REF-001",
            )]

        # ACM uses author-number format: [Author Year] or [1]
        # Detect inconsistent styles
        has_numeric = any(re.match(r"^\[\d+\]", r) for r in doc.references)
        has_name = any(re.match(r"^\[[A-Z]", r) for r in doc.references)
        if has_numeric and has_name:
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Mixed reference styles detected. ACM uses a single consistent style.",
                suggestion="Use ACM's reference format consistently (see ACM Master Template).",
                rule_id="ACM-REF-002",
            ))

        # Check for missing DOIs
        missing_doi = sum(
            1 for r in doc.references if not re.search(r"doi|10\.\d{4}|https?://dl\.acm", r, re.I)
        )
        if missing_doi > len(doc.references) * 0.3:
            findings.append(Finding(
                category="style",
                severity="info",
                message=f"{missing_doi} of {len(doc.references)} references appear to be missing DOIs.",
                suggestion="ACM strongly recommends DOIs for all references.",
                rule_id="ACM-REF-003",
            ))
        return findings

    def _check_author_info(self, doc: Document) -> list[Finding]:
        findings = []
        text = doc.raw_text
        # ORCID presence check
        if not re.search(r"orcid|0000-\d{4}-\d{4}-\d{4}", text, re.I):
            findings.append(Finding(
                category="style",
                severity="info",
                message="ORCID iD not detected. ACM encourages authors to include ORCID identifiers.",
                suggestion="Register at https://orcid.org and include your 16-digit ORCID iD.",
                rule_id="ACM-AUTH-001",
            ))
        # Author affiliation
        if not re.search(r"university|institute|corporation|laboratory|department", text, re.I):
            findings.append(Finding(
                category="style",
                severity="warning",
                message="Author affiliation not detected.",
                suggestion="Include institution, city, country, and email for each author.",
                rule_id="ACM-AUTH-002",
            ))
        return findings

    def _check_artifact_availability(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_artifact = any(k in text_lower for k in [
            "github", "gitlab", "zenodo", "figshare", "artifact", "source code",
            "data availability", "reproducib", "open source", "replicab"
        ])
        if not has_artifact:
            return [Finding(
                category="style",
                severity="info",
                message="No artifact availability or reproducibility statement detected.",
                suggestion="ACM encourages artifact evaluation. Link code/data repositories for reproducibility badges.",
                rule_id="ACM-ART-001",
            )]
        return []

    def _check_rights(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        has_rights = any(k in text_lower for k in [
            "acm", "permission to make", "copyright", "creative commons", "cc by"
        ])
        if not has_rights:
            return [Finding(
                category="style",
                severity="info",
                message="ACM rights/license block not detected (required in camera-ready submissions).",
                suggestion="Include the ACM rights statement generated by the rights management system.",
                rule_id="ACM-RIGHTS-001",
            )]
        return []

    def _check_ethics(self, doc: Document) -> list[Finding]:
        text_lower = doc.raw_text.lower()
        # For CHI, FAccT, CSCW -- ethics review is mandatory
        has_ethics = any(k in text_lower for k in [
            "ethical", "irb", "institutional review", "informed consent",
            "participants", "human subjects"
        ])
        mentions_humans = any(k in text_lower for k in [
            "user study", "participants", "survey", "interview", "human"
        ])
        if mentions_humans and not has_ethics:
            return [Finding(
                category="style",
                severity="warning",
                message="Human subjects study detected but no ethics/IRB statement found.",
                suggestion="ACM requires ethics approval statements for research involving human participants.",
                rule_id="ACM-ETH-001",
            )]
        return []

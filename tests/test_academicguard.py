"""Basic smoke tests for AcademicGuard."""

import pytest
from academicguard.core.document import Document
from academicguard.detectors.ai_detector import AIDetector
from academicguard.detectors.grammar import GrammarChecker, flesch_kincaid_grade
from academicguard.style import get_style_checker, VENUE_REGISTRY

SAMPLE_TEXT = """
Abstract-- This paper presents a novel framework for network intrusion detection using
machine learning techniques. The proposed approach achieves a detection accuracy of 98.7%
on the NSL-KDD dataset with a false positive rate below 0.3%. Experimental results
demonstrate the effectiveness of the proposed method compared to state-of-the-art baselines.

Keywords-- intrusion detection, machine learning, deep learning, network security

I. Introduction

Network intrusion detection systems (NIDS) play a critical role in modern cybersecurity
infrastructure. With the proliferation of sophisticated attacks, traditional signature-based
methods struggle to detect zero-day exploits. In this paper, we propose a hybrid approach
combining supervised and unsupervised learning to address these limitations.

The main contributions of this work are:
1. A novel feature selection algorithm reducing dimensionality by 60%.
2. A hybrid detection model achieving superior accuracy.
3. Evaluation on three publicly available benchmark datasets.

II. Related Work

Smith et al. [1] proposed a decision-tree-based approach achieving 95% accuracy.
Jones and Brown [2] extended this work using ensemble methods. Our approach differs
by incorporating deep learning for feature extraction.

III. Methodology

Our system consists of three components: preprocessing, feature extraction, and
classification. We evaluate performance using standard metrics including precision,
recall, F1-score, and AUC-ROC.

IV. Results

Table I shows the comparative performance. Our method achieves 98.7% accuracy,
outperforming all baselines by a statistically significant margin (p < 0.01).

V. Conclusion

We presented a novel intrusion detection framework achieving state-of-the-art performance.
Future work will extend the system to real-time streaming environments.

References
[1] J. Smith et al., "Decision tree intrusion detection," IEEE Trans. Inf. Forensics, 2022.
[2] A. Jones and B. Brown, "Ensemble methods for NIDS," Comput. Secur., 2023.
"""


class TestDocumentParsing:
    def test_from_string(self):
        doc = Document.from_string(SAMPLE_TEXT, title="Test Paper")
        assert doc.word_count > 100
        assert doc.title == "Test Paper"

    def test_abstract_detection(self):
        doc = Document.from_string(SAMPLE_TEXT)
        assert len(doc.abstract) > 0

    def test_keyword_detection(self):
        doc = Document.from_string(SAMPLE_TEXT)
        assert len(doc.keywords) >= 2

    def test_section_detection(self):
        doc = Document.from_string(SAMPLE_TEXT)
        assert len(doc.sections) >= 2

    def test_reference_detection(self):
        doc = Document.from_string(SAMPLE_TEXT)
        assert len(doc.references) >= 1


class TestAIDetector:
    def test_ai_detector_runs(self):
        doc = Document.from_string(SAMPLE_TEXT)
        detector = AIDetector(use_transformer=False)
        result = detector.analyze(doc)
        assert result.module == "AI Detector"
        assert 0.0 <= result.score <= 1.0
        assert result.label in {"PASS", "WARN", "FAIL"}

    def test_burstiness_on_human_text(self):
        detector = AIDetector(use_transformer=False)
        result = detector._detect(SAMPLE_TEXT)
        assert 0.0 <= result.burstiness_score <= 1.0

    def test_repetition_signal(self):
        ai_text = (
            "Furthermore, it is important to note that the proposed novel approach "
            "leverages state-of-the-art techniques. Moreover, the results demonstrate "
            "significant improvements. To summarize, this paper presents a comprehensive "
            "framework. In conclusion, the methodology is robust and seamlessly integrates."
        )
        detector = AIDetector(use_transformer=False)
        result = detector._detect(ai_text)
        assert result.repetition_score > 0.2  # should detect AI phrases


class TestGrammarChecker:
    def test_grammar_runs(self):
        doc = Document.from_string(SAMPLE_TEXT)
        gc = GrammarChecker()
        result = gc.analyze(doc)
        assert result.module == "Grammar"
        assert 0.0 <= result.score <= 1.0
        gc.close()

    def test_contraction_detection(self):
        text = "We don't use contractions in academic writing. It's not appropriate."
        doc = Document.from_string(text)
        gc = GrammarChecker()
        findings = gc._check_register(text)
        contraction_findings = [f for f in findings if f.rule_id == "REG-001"]
        assert len(contraction_findings) >= 2
        gc.close()

    def test_fk_grade_range(self):
        grade = flesch_kincaid_grade(SAMPLE_TEXT)
        assert 5.0 <= grade <= 25.0  # reasonable range for academic text


class TestStyleCheckers:
    @pytest.mark.parametrize("venue", list(VENUE_REGISTRY.keys()))
    def test_all_venues_run(self, venue):
        doc = Document.from_string(SAMPLE_TEXT)
        checker = get_style_checker(venue)
        result = checker.analyze(doc)
        assert result.module == "Style"
        assert 0.0 <= result.score <= 1.0
        assert result.label in {"PASS", "WARN", "FAIL"}

    def test_ieee_citation_detection(self):
        from academicguard.style.ieee import IEEEStyleChecker
        doc = Document.from_string(SAMPLE_TEXT)
        checker = IEEEStyleChecker()
        findings = checker._check(doc)
        assert isinstance(findings, list)

    def test_unknown_venue_raises(self):
        with pytest.raises(ValueError):
            get_style_checker("nonexistent_venue_xyz")

    def test_elsevier_highlights_missing(self):
        from academicguard.style.elsevier import ElsevierStyleChecker
        doc = Document.from_string(SAMPLE_TEXT)
        checker = ElsevierStyleChecker()
        findings = checker._check(doc)
        highlight_findings = [f for f in findings if f.rule_id == "ELS-HLT-001"]
        assert len(highlight_findings) == 1  # highlights are missing


class TestReport:
    def test_report_json(self):
        import json
        from academicguard.core.report import AnalysisReport, ModuleResult, Finding
        report = AnalysisReport(document_title="Test", venue="IEEE")
        report.modules.append(ModuleResult(
            module="AI Detector", score=0.9, label="PASS", summary="Test",
            findings=[Finding(category="ai", severity="info", message="Test finding")]
        ))
        report.compute_overall()
        data = json.loads(report.to_json())
        assert data["overall_label"] in {"PASS", "WARN", "FAIL"}
        assert len(data["modules"]) == 1

    def test_report_html(self):
        from academicguard.core.report import AnalysisReport
        report = AnalysisReport(document_title="Test Paper", venue="IEEE")
        report.compute_overall()
        html = report._render_html()
        assert "AcademicGuard" in html
        assert "Test Paper" in html

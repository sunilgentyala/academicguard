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

    def test_signals_present(self):
        detector = AIDetector(use_transformer=False)
        result = detector._detect(SAMPLE_TEXT)
        signal_names = {s.name for s in result.signals}
        assert "Burstiness" in signal_names
        assert "Yule K" in signal_names
        assert "Hapax Rate" in signal_names
        assert "N-gram Entropy" in signal_names
        assert "Stylometric Profile" in signal_names
        assert "Semantic Coherence" in signal_names

    def test_burstiness_value(self):
        from academicguard.detectors.ai_detector import BurstinessAnalyzer
        ba = BurstinessAnalyzer()
        burst = ba.analyze(SAMPLE_TEXT)
        assert 0.0 <= burst <= 5.0

    def test_yule_k(self):
        from academicguard.detectors.ai_detector import YuleKAnalyzer
        ya = YuleKAnalyzer()
        K = ya.analyze(SAMPLE_TEXT)
        assert K >= 0.0

    def test_hapax_rate(self):
        from academicguard.detectors.ai_detector import HapaxAnalyzer
        ha = HapaxAnalyzer()
        rate = ha.analyze(SAMPLE_TEXT)
        assert 0.0 <= rate <= 1.0

    def test_zipf(self):
        from academicguard.detectors.ai_detector import ZipfAnalyzer
        za = ZipfAnalyzer()
        r2 = za.analyze(SAMPLE_TEXT)
        assert 0.0 <= r2 <= 1.0

    def test_ngram_entropy(self):
        from academicguard.detectors.ai_detector import NGramEntropyAnalyzer
        na = NGramEntropyAnalyzer()
        stats = na.analyze(SAMPLE_TEXT)
        assert "word_bigram" in stats
        assert "char_trigram" in stats
        assert stats["word_bigram"] > 0

    def test_gltr_heuristic(self):
        from academicguard.detectors.ai_detector import GLTRAnalyzer
        ga = GLTRAnalyzer()
        result = ga._heuristic_gltr(SAMPLE_TEXT)
        assert "green_fraction" in result
        assert 0.0 <= result["green_fraction"] <= 1.0

    def test_semantic_coherence(self):
        from academicguard.detectors.ai_detector import SemanticCoherenceAnalyzer
        sca = SemanticCoherenceAnalyzer()
        coherence = sca.analyze(SAMPLE_TEXT)
        assert 0.0 <= coherence <= 1.0

    def test_ai_phrase_detection(self):
        ai_text = (
            "Furthermore, it is important to note that the proposed novel approach "
            "leverages state-of-the-art techniques seamlessly. Moreover, the experimental "
            "results demonstrate significant improvements. In conclusion, this comprehensive "
            "framework is robust and outperforms all existing baselines."
        )
        detector = AIDetector(use_transformer=False)
        result = detector._detect(ai_text)
        stylo_signal = next(s for s in result.signals if s.name == "Stylometric Profile")
        assert stylo_signal.ai_probability > 0.3


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

class TestPlagiarismDetector:
    def test_winnowing_fingerprint(self):
        from academicguard.detectors.plagiarism import WinnowingFingerprinter
        wp = WinnowingFingerprinter()
        fp1 = wp.fingerprint("The quick brown fox jumps over the lazy dog")
        fp2 = wp.fingerprint("The quick brown fox jumps over the lazy dog")
        fp3 = wp.fingerprint("Completely different text about machine learning")
        assert wp.similarity(fp1, fp2) > 0.9
        assert wp.similarity(fp1, fp3) < 0.5

    def test_tfidf_similarity(self):
        from academicguard.detectors.plagiarism import TFIDFSimilarityChecker
        tc = TFIDFSimilarityChecker()
        sim, _ = tc.sentence_similarity(
            "Machine learning algorithms detect intrusions effectively.",
            "Machine learning algorithms detect intrusions effectively.",
        )
        assert sim > 0.8

    def test_plagiarism_detector_no_corpus(self):
        doc = Document.from_string(SAMPLE_TEXT)
        pd = PlagiarismDetector()
        result = pd.analyze(doc)
        assert result.module == "Plagiarism"
        assert 0.0 <= result.score <= 1.0
        assert "Winnowing" in result.metadata["algorithms_used"][0]


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

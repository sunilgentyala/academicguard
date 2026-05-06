"""AcademicGuard high-level Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import AnalysisReport
from academicguard.detectors.ai_detector import AIDetector
from academicguard.detectors.plagiarism import PlagiarismDetector
from academicguard.detectors.grammar import GrammarChecker
from academicguard.style import get_style_checker


def analyze(
    source: str | Path,
    venue: str = "ieee",
    corpus_dir: Optional[str | Path] = None,
    use_transformer: bool = True,
    run_ai: bool = True,
    run_plagiarism: bool = True,
    run_grammar: bool = True,
    run_style: bool = True,
) -> AnalysisReport:
    """
    Full analysis pipeline.

    Parameters
    ----------
    source : str | Path
        File path (.txt, .docx, .pdf, .tex) or raw text string.
    venue : str
        Target venue key: 'ieee', 'elsevier', 'acm', 'iet', 'bcs'.
    corpus_dir : optional path
        Local directory of .txt reference files for plagiarism comparison.
    use_transformer : bool
        If True, load GPT-2 for perplexity-based AI detection (slower but better).
    run_* : bool
        Toggle individual modules.

    Returns
    -------
    AnalysisReport
        Populated report with scores and findings for each module.

    Example
    -------
    >>> from academicguard import api
    >>> report = api.analyze("paper.pdf", venue="ieee")
    >>> print(report.overall_label, report.overall_score)
    >>> report.save_html("report.html")
    """
    # Load document
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        doc = Document.from_file(source)
    else:
        doc = Document.from_string(str(source))

    report = AnalysisReport(
        document_title=doc.title or str(source)[:80],
        venue=venue.upper(),
    )

    if run_ai:
        detector = AIDetector(use_transformer=use_transformer)
        report.modules.append(detector.analyze(doc))

    if run_plagiarism:
        pd = PlagiarismDetector(corpus_dir=corpus_dir)
        report.modules.append(pd.analyze(doc))

    if run_grammar:
        gc = GrammarChecker()
        report.modules.append(gc.analyze(doc))
        gc.close()

    if run_style:
        checker = get_style_checker(venue)
        report.modules.append(checker.analyze(doc))

    report.compute_overall()
    return report


def analyze_text(
    text: str,
    venue: str = "ieee",
    **kwargs,
) -> AnalysisReport:
    """Analyze a raw text string directly."""
    return analyze(text, venue=venue, **kwargs)

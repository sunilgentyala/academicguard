"""Detectors package init."""
from academicguard.detectors.ai_detector import AIDetector
from academicguard.detectors.plagiarism import PlagiarismDetector
from academicguard.detectors.grammar import GrammarChecker

__all__ = ["AIDetector", "PlagiarismDetector", "GrammarChecker"]

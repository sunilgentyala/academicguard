"""AcademicGuard: Open-source academic writing integrity toolkit."""

__version__ = "1.0.0"
__author__ = "AcademicGuard Contributors"
__license__ = "MIT"

from academicguard.core.document import Document
from academicguard.core.report import AnalysisReport

__all__ = ["Document", "AnalysisReport"]

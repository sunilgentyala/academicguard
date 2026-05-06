"""Core package init."""
from academicguard.core.document import Document, DocumentSection
from academicguard.core.report import AnalysisReport, Finding, ModuleResult

__all__ = ["Document", "DocumentSection", "AnalysisReport", "Finding", "ModuleResult"]

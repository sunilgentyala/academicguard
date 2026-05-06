"""Unified analysis report: collects findings from all detectors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, BaseLoader


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class Finding:
    category: str           # "grammar", "ai", "plagiarism", "style"
    severity: str           # "error", "warning", "info"
    message: str
    location: str = ""      # "Section 2, paragraph 3" or "Line 45"
    suggestion: str = ""
    rule_id: str = ""
    context: str = ""       # short snippet of problematic text


@dataclass
class ModuleResult:
    module: str
    score: float            # 0.0–1.0 (1.0 = perfect / no issues)
    label: str              # "PASS", "WARN", "FAIL"
    summary: str
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    document_title: str
    venue: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modules: list[ModuleResult] = field(default_factory=list)
    overall_score: float = 0.0
    overall_label: str = "UNKNOWN"

    # ------------------------------------------------------------------ #
    # Score aggregation
    # ------------------------------------------------------------------ #

    def compute_overall(self) -> None:
        if not self.modules:
            return
        weights = {
            "AI Detector": 0.25,
            "Plagiarism": 0.30,
            "Grammar": 0.20,
            "Style": 0.25,
        }
        total_w = 0.0
        weighted_sum = 0.0
        for m in self.modules:
            w = weights.get(m.module, 0.15)
            weighted_sum += m.score * w
            total_w += w
        self.overall_score = weighted_sum / total_w if total_w else 0.0
        if self.overall_score >= 0.85:
            self.overall_label = "PASS"
        elif self.overall_score >= 0.65:
            self.overall_label = "WARN"
        else:
            self.overall_label = "FAIL"

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "document_title": self.document_title,
            "venue": self.venue,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 3),
            "overall_label": self.overall_label,
            "modules": [
                {
                    "module": m.module,
                    "score": round(m.score, 3),
                    "label": m.label,
                    "summary": m.summary,
                    "metadata": m.metadata,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "message": f.message,
                            "location": f.location,
                            "suggestion": f.suggestion,
                            "rule_id": f.rule_id,
                            "context": f.context,
                        }
                        for f in m.findings
                    ],
                }
                for m in self.modules
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def save_html(self, path: str | Path) -> None:
        Path(path).write_text(self._render_html(), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # HTML rendering
    # ------------------------------------------------------------------ #

    def _render_html(self) -> str:
        env = Environment(loader=BaseLoader(), autoescape=True)
        tmpl = env.from_string(_HTML_TEMPLATE)
        return tmpl.render(report=self, data=self.to_dict())

    def summary_text(self) -> str:
        lines = [
            f"AcademicGuard Analysis Report",
            f"Document : {self.document_title}",
            f"Venue    : {self.venue}",
            f"Date     : {self.timestamp[:10]}",
            f"Overall  : {self.overall_label} ({self.overall_score:.1%})",
            "",
        ]
        for m in self.modules:
            lines.append(f"  [{m.label:4}] {m.module:20} {m.score:.1%}  {m.summary}")
            for f in m.findings[:5]:
                lines.append(f"         [{f.severity.upper():5}] {f.message[:90]}")
            if len(m.findings) > 5:
                lines.append(f"         ... and {len(m.findings)-5} more findings")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# HTML template
# ------------------------------------------------------------------ #

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AcademicGuard Report -- {{ report.document_title }}</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}
  h1{color:#1a3a6b}h2{color:#2c5282;border-bottom:2px solid #bee3f8;padding-bottom:.3rem}
  .badge{display:inline-block;padding:.25rem .65rem;border-radius:4px;font-weight:700;font-size:.85rem}
  .PASS{background:#c6f6d5;color:#22543d}.WARN{background:#fefcbf;color:#744210}.FAIL{background:#fed7d7;color:#742a2a}
  .score-bar{height:12px;border-radius:6px;background:#e2e8f0;margin:.4rem 0}
  .score-fill{height:100%;border-radius:6px;background:#4299e1}
  table{width:100%;border-collapse:collapse;margin:1rem 0}
  th{background:#2c5282;color:#fff;padding:.5rem}td{padding:.45rem;border-bottom:1px solid #e2e8f0}
  .error{color:#c53030}.warning{color:#c05621}.info{color:#2b6cb0}
  .module-card{border:1px solid #bee3f8;border-radius:8px;padding:1rem;margin:1rem 0}
  footer{margin-top:3rem;font-size:.8rem;color:#718096;text-align:center}
</style>
</head>
<body>
<h1>AcademicGuard Analysis Report</h1>
<table>
<tr><th>Document</th><td>{{ report.document_title }}</td><th>Venue</th><td>{{ report.venue }}</td></tr>
<tr><th>Date</th><td>{{ report.timestamp[:10] }}</td><th>Overall</th>
<td><span class="badge {{ report.overall_label }}">{{ report.overall_label }} ({{ "%.1f"|format(report.overall_score*100) }}%)</span></td></tr>
</table>

{% for m in data.modules %}
<div class="module-card">
<h2>{{ m.module }} <span class="badge {{ m.label }}">{{ m.label }} {{ "%.1f"|format(m.score*100) }}%</span></h2>
<div class="score-bar"><div class="score-fill" style="width:{{ (m.score*100)|int }}%"></div></div>
<p>{{ m.summary }}</p>
{% if m.findings %}
<table>
<tr><th>Severity</th><th>Message</th><th>Location</th><th>Suggestion</th></tr>
{% for f in m.findings %}
<tr>
<td class="{{ f.severity }}">{{ f.severity.upper() }}</td>
<td>{{ f.message }}</td>
<td>{{ f.location }}</td>
<td>{{ f.suggestion }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p><em>No findings.</em></p>
{% endif %}
</div>
{% endfor %}

<footer>Generated by <strong>AcademicGuard v1.0.0</strong> -- Open Source Academic Writing Integrity Toolkit</footer>
</body>
</html>
"""


# AcademicGuard -- How-To Reference Guide

**Version 1.0.0 | Open Source | MIT License**

AcademicGuard is a fully offline-capable, open-source toolkit for academic writing
integrity checking. It combines AI content detection, plagiarism analysis, English
grammar/register checking, and venue-specific style enforcement in a single unified tool.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [CLI Reference](#3-cli-reference)
4. [Python API](#4-python-api)
5. [AI Content Detection](#5-ai-content-detection)
6. [Plagiarism Detection](#6-plagiarism-detection)
7. [Grammar and Academic English](#7-grammar-and-academic-english)
8. [Venue Style Checkers](#8-venue-style-checkers)
   - [IEEE](#ieee)
   - [Elsevier](#elsevier)
   - [ACM](#acm)
   - [IET](#iet)
   - [BCS](#bcs)
9. [External Service Integrations](#9-external-service-integrations)
10. [Report Formats](#10-report-formats)
11. [Environment Variables](#11-environment-variables)
12. [Supported File Formats](#12-supported-file-formats)
13. [Extending AcademicGuard](#13-extending-academicguard)
14. [Troubleshooting](#14-troubleshooting)
15. [Industry Tool Comparison](#15-industry-tool-comparison)
16. [Venue Compliance Checklists](#16-venue-compliance-checklists)

---

## 1. Installation

### Requirements
- Python 3.10 or later
- 2 GB disk space (for GPT-2 model download on first run)
- Internet connection (first run only for model/LanguageTool download)

### Install from PyPI
```bash
pip install academicguard
```

### Install from Source
```bash
git clone https://github.com/academicguard/academicguard.git
cd academicguard
pip install -e ".[dev]"
```

### Install spaCy language model (required for grammar)
```bash
python -m spacy download en_core_web_sm
```

### Optional: Install LanguageTool locally (recommended for privacy)
```bash
# Docker (recommended)
docker run -d -p 8010:8010 silviof/docker-languagetool
export LANGUAGETOOL_URL="http://localhost:8010"

# Or via Java (requires JRE 17+)
# Download from: https://languagetool.org/download/
```

### Optional: PDF export support
```bash
pip install "academicguard[pdf-export]"
```

---

## 2. Quick Start

```bash
# Analyze a PDF paper for IEEE submission
academicguard analyze paper.pdf --venue ieee

# Save an HTML report
academicguard analyze paper.pdf --venue elsevier --html report.html

# Analyze a LaTeX file for ACM submission
academicguard analyze paper.tex --venue acm --json report.json

# Run only AI detection (fastest)
academicguard ai paper.pdf

# Run only grammar check
academicguard grammar paper.docx
```

### Python API Quick Start
```python
import academicguard.api as ag

# Full analysis
report = ag.analyze("paper.pdf", venue="ieee")
print(f"Result: {report.overall_label} ({report.overall_score:.0%})")

# Save reports
report.save_html("report.html")
report.save_json("report.json")

# Check individual findings
for module in report.modules:
    print(f"\n{module.module}: {module.label}")
    for finding in module.findings:
        print(f"  [{finding.severity.upper()}] {finding.message}")
```

---

## 3. CLI Reference

### `academicguard analyze`

The primary command. Runs all detection modules.

```
academicguard analyze FILE [OPTIONS]

Arguments:
  FILE    Path to document (.txt, .docx, .pdf, .tex)

Options:
  --venue, -v        Target venue [ieee|elsevier|acm|iet|bcs]  (default: ieee)
  --json, -j PATH    Save JSON report to PATH
  --html, -H PATH    Save HTML report to PATH
  --corpus, -c DIR   Local corpus directory for plagiarism comparison
  --skip-ai          Skip AI detection
  --skip-plagiarism  Skip plagiarism check
  --skip-grammar     Skip grammar check
  --skip-style       Skip venue style check
  --no-transformer   Use heuristic AI detection (no GPT-2, much faster)
  --verbose, -V      Show info-level findings in addition to errors/warnings
```

**Exit codes:**
- `0` -- PASS (all modules passed)
- `1` -- WARN (warnings present, no hard errors)
- `2` -- FAIL (one or more hard errors)

### Sub-commands

| Command | Description |
|---------|-------------|
| `academicguard ai FILE` | AI detection only |
| `academicguard plagiarism FILE [--corpus DIR]` | Plagiarism check only |
| `academicguard grammar FILE [--lang en-US]` | Grammar check only |
| `academicguard style FILE --venue VENUE` | Style check only |
| `academicguard venues` | List all supported venues |
| `academicguard services` | Show external service status |
| `academicguard setup-env` | Print env var configuration guide |
| `academicguard version` | Print version |

### Examples

```bash
# Fast check (no GPT-2, no grammar server)
academicguard analyze paper.pdf --venue ieee --no-transformer --skip-grammar

# IEEE conference submission check
academicguard analyze conference_paper.pdf --venue ieee --html ieee_report.html

# Elsevier journal check with competing interests verification
academicguard analyze journal_paper.docx --venue elsevier --json elsevier_report.json

# BCS with local corpus for self-plagiarism detection
academicguard analyze bcs_paper.txt --venue bcs --corpus ~/my_previous_papers/

# Check against multiple venues (shell loop)
for venue in ieee acm iet; do
  academicguard analyze paper.pdf --venue $venue --json report_$venue.json
done

# CI/CD integration (GitHub Actions)
academicguard analyze paper.pdf --venue ieee --no-transformer
# Exit code 2 = FAIL, blocks pipeline
```

---

## 4. Python API

### `academicguard.api.analyze()`

```python
def analyze(
    source: str | Path,         # file path or raw text string
    venue: str = "ieee",        # target venue
    corpus_dir: Path = None,    # local corpus for plagiarism
    use_transformer: bool = True,  # use GPT-2 for AI detection
    run_ai: bool = True,
    run_plagiarism: bool = True,
    run_grammar: bool = True,
    run_style: bool = True,
) -> AnalysisReport
```

### Working with `AnalysisReport`

```python
report = ag.analyze("paper.pdf", venue="ieee")

# Overall result
report.overall_label   # "PASS", "WARN", or "FAIL"
report.overall_score   # float 0.0-1.0
report.timestamp       # ISO 8601 timestamp

# Per-module results
for module in report.modules:
    module.module   # "AI Detector", "Plagiarism", "Grammar", "Style"
    module.score    # 0.0-1.0
    module.label    # "PASS", "WARN", "FAIL"
    module.summary  # human-readable summary string
    module.metadata # dict with detailed metrics

    for finding in module.findings:
        finding.severity    # "error", "warning", "info"
        finding.message     # description
        finding.suggestion  # remediation advice
        finding.rule_id     # e.g., "IEEE-CITE-001"
        finding.location    # "Section 3, offset 1242"
        finding.context     # short text snippet

# Export
report.save_html("report.html")
report.save_json("report.json")
report.to_dict()    # plain dict
report.to_json()    # JSON string
report.summary_text()  # plain text summary
```

### Using individual modules

```python
from academicguard.core.document import Document
from academicguard.detectors.ai_detector import AIDetector
from academicguard.detectors.plagiarism import PlagiarismDetector
from academicguard.detectors.grammar import GrammarChecker
from academicguard.style import get_style_checker

doc = Document.from_file("paper.pdf")

# AI detection
detector = AIDetector(use_transformer=True)
ai_result = detector.analyze(doc)
print(f"AI probability: {ai_result.metadata['overall_ai_probability']:.0%}")

# Plagiarism
pd = PlagiarismDetector(corpus_dir="/path/to/corpus")
plag_result = pd.analyze(doc)
print(f"Max similarity: {plag_result.metadata['overall_similarity']:.0%}")

# Grammar
gc = GrammarChecker(language="en-US")
gram_result = gc.analyze(doc)
print(f"FK Grade: {gram_result.metadata['fk_grade_level']}")
gc.close()

# Style
checker = get_style_checker("ieee")
style_result = checker.analyze(doc)

# Text-only AI detection
result = detector.analyze_text("Your raw text here...")
print(result.label, result.overall_probability)
```

---

## 5. AI Content Detection

### How It Works

AcademicGuard uses a four-signal ensemble for AI detection -- no external API required:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| **Perplexity** | 35% | GPT-2 surprise at each token. AI text is more predictable (lower perplexity). |
| **Burstiness** | 25% | Coefficient of variation in sentence lengths. Humans are "bursty"; AI is uniform. |
| **Repetition** | 25% | Density of known AI-generated filler phrases ("furthermore", "delve", "leveraging", etc.). |
| **Vocabulary richness** | 15% | Type-token ratio and hapax legomena rate. AI text is lexically narrower. |

### Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| AI probability >= 75% | `LIKELY_AI` | High confidence of AI generation |
| 45-75% | `UNCERTAIN` | Mixed signals; manual review recommended |
| < 45% | `LIKELY_HUMAN` | Predominantly human-authored |

### Configuration

```bash
# Use GPT-2 (best accuracy, ~500MB download)
academicguard ai paper.pdf

# Heuristic mode (no download, faster, less accurate)
academicguard ai paper.pdf --no-transformer

# Enable ZeroGPT as second opinion
export ZEROGPT_API_KEY="your_key"
academicguard analyze paper.pdf  # uses ZeroGPT automatically if key is set

# Enable GPTZero as second opinion
export GPTZERO_API_KEY="your_key"
```

### Known AI Phrase Database

The tool flags these common AI-generated patterns:
- Discourse markers: "Furthermore", "Moreover", "In conclusion", "To summarize"
- Epistemic hedges (overused): "It is important to note", "It is worth noting"
- AI-typical adjectives: "robust", "seamless", "comprehensive", "novel approach"
- AI-typical verbs: "leverage", "delve", "demonstrate"
- Generic claims: "The results show significant improvement"

### Limitations

- Cannot detect AI text that has been substantially edited by a human
- Short texts (< 100 words) produce unreliable scores
- Domain-specific jargon may inflate perplexity scores
- Works best on English text

---

## 6. Plagiarism Detection

### Detection Layers

1. **Local corpus (MinHash/LSH)** -- compare against your own document collection
2. **CrossRef metadata** -- detect duplicate titles in academic databases (free, no API key)
3. **Turnitin iThenticate** -- institutional-grade similarity detection (requires license)
4. **Copyscape Premium** -- web-based duplicate detection (pay-per-search)

### Local Corpus Setup

```bash
# Create a corpus directory with reference documents
mkdir ~/corpus
cp ~/previous_papers/*.txt ~/corpus/
cp ~/sources/*.txt ~/corpus/

# Run with corpus
academicguard plagiarism paper.txt --corpus ~/corpus/
```

Similarity thresholds:
- `>= 80%` -- Error: clear duplication, citation required
- `40-80%` -- Warning: substantial overlap, review needed
- `< 40%` -- Info: minor overlap, likely acceptable

### External API Configuration

```bash
# Turnitin iThenticate
export TURNITIN_API_KEY="your_api_key"

# Copyscape
export COPYSCAPE_USER="your_username"
export COPYSCAPE_KEY="your_api_key"
```

### Self-Plagiarism Check

Supply your own previous papers as the corpus to detect self-plagiarism:

```bash
academicguard plagiarism new_paper.pdf --corpus ~/all_my_papers/
```

---

## 7. Grammar and Academic English

### What Is Checked

| Rule Category | Examples |
|--------------|---------|
| **Contractions** | "don't" -> "do not"; "it's" -> "it is" |
| **Colloquial language** | "a lot of", "kind of", "basically", "stuff" |
| **Vague intensifiers** | "very important" -> quantify with numbers |
| **Wordiness** | "in order to" -> "to"; "due to the fact that" -> "because" |
| **First-person singular** | Flag excessive "I" vs. "we" |
| **Hedging absence** | Warn if no epistemic hedging in abstract |
| **LanguageTool rules** | 5000+ grammar/spelling/style rules |
| **Readability** | Flesch-Kincaid grade (target 12-16) |
| **Sentence length** | Flag > 40 words or < 10 words average |
| **Passive voice** | Count and flag excessive passive constructions |

### Readability Targets for Academic Writing

| Metric | Target Range | Meaning |
|--------|-------------|---------|
| Flesch-Kincaid Grade | 12-16 | Graduate-level academic text |
| Flesch Reading Ease | 30-50 | Formal/academic register |
| Average Sentence Length | 18-30 words | Clear but complex |

### Using British vs. American English

```bash
# American English (IEEE, Elsevier en-US)
academicguard grammar paper.pdf --lang en-US

# British English (IET, BCS, Oxford journals)
academicguard grammar paper.pdf --lang en-GB
```

The IET and BCS checkers also flag British vs. American spelling differences automatically.

### LanguageTool Setup

LanguageTool is downloaded automatically on first use. For privacy or performance:

```bash
# Self-hosted Docker
docker run -d -p 8010:8010 silviof/docker-languagetool
export LANGUAGETOOL_URL="http://localhost:8010"

# Premium API (more rules)
export LANGUAGETOOL_URL="https://api.languagetool.org"
```

---

## 8. Venue Style Checkers

### IEEE

**Applies to:** IEEE Transactions, IEEE Access, IEEE Letters, IEEE Conference Proceedings (ICASSP, INFOCOM, ICC, GLOBECOM, etc.)

**Reference:** [IEEE Author Center](https://ieeeauthorcenter.ieee.org)

#### Rules Checked

| Rule ID | Category | Requirement |
|---------|----------|-------------|
| IEEE-ABS-001 | Abstract | 150-250 words |
| IEEE-KW-001 | Index Terms | 3-10 terms |
| IEEE-KW-002 | Index Terms | Lowercase unless proper noun/acronym |
| IEEE-TTL-001 | Title | Maximum 15 words |
| IEEE-TTL-002 | Title | Title case (capitalize major words) |
| IEEE-SEC-001 | Sections | Introduction and Conclusion required |
| IEEE-SEC-002 | Sections | Related Work recommended |
| IEEE-SEC-003 | Sections | Consistent Roman or Arabic numbering |
| IEEE-CITE-001 | Citations | Use [N] numeric style, not author-year |
| IEEE-CITE-002 | Citations | Numbered in order of first appearance |
| IEEE-UNIT-001 | Units | Space between number and SI unit (e.g., "5 GHz") |
| IEEE-ETAL-001 | References | Correct "et al." formatting |
| IEEE-FIG-001 | Figures | "Fig. N" abbreviated in body text |
| IEEE-ACR-001 | Acronyms | Define on first use; skip if used once |

#### IEEE Reference Format

```
[1] J. A. Smith, B. C. Jones, and D. E. Brown,
    "Title of the paper in sentence case,"
    IEEE Trans. Inf. Forensics Security, vol. 18, pp. 1234-1250, 2023.
    doi: 10.1109/TIFS.2023.xxxxxxx
```

---

### Elsevier

**Applies to:** All ScienceDirect journals (Computers & Security, Expert Systems with Applications, Information Sciences, Pattern Recognition, Knowledge-Based Systems, etc.)

**Reference:** [Elsevier Author Guidelines](https://www.elsevier.com/authors)

#### Rules Checked

| Rule ID | Category | Requirement |
|---------|----------|-------------|
| ELS-ABS-001 | Abstract | 150-300 words |
| ELS-ABS-002 | Abstract | Structured abstract recommended for long papers |
| ELS-KW-001 | Keywords | 4-8 keywords |
| ELS-KW-002 | Keywords | No trailing punctuation |
| ELS-HLT-001 | Highlights | 3-5 bullet points required (max 85 chars each) |
| ELS-HLT-002 | Highlights | Each highlight <= 85 characters |
| ELS-CREDIT-001 | Authors | CRediT Author Contribution Statement required |
| ELS-REF-001 | References | References section required |
| ELS-REF-002 | References | DOI recommended for all references |
| ELS-SEC-001 | Sections | IMRAD structure (Intro, Methods, Results, Discussion, Conclusion) |
| ELS-ETH-001 | Ethics | Funding/Data Availability/Ethics statements required |
| ELS-FIG-001 | Figures | Self-explanatory captions (>= 5 words) |
| ELS-COI-001 | Compliance | Declaration of Competing Interests required |

#### Elsevier Highlights Format

```
Highlights
• Novel deep learning architecture for network intrusion detection [< 85 chars]
• Achieves 98.7% accuracy on NSL-KDD benchmark dataset [< 85 chars]
• Reduces false positive rate by 40% compared to baselines [< 85 chars]
• Validated on three real-world network traffic datasets [< 85 chars]
```

#### CRediT Statement Example

```
Author Contributions:
Alice Smith: Conceptualization, Methodology, Software, Formal analysis, Writing -- original draft.
Bob Jones: Data curation, Validation, Writing -- review and editing.
Carol Brown: Supervision, Funding acquisition, Project administration.
```

---

### ACM

**Applies to:** ACM Digital Library publications, SIGCOMM, CCS, SOSP, CHI, FAccT, CSCW, USENIX (affiliated), etc.

**Reference:** [ACM Author Center](https://authors.acm.org)

#### Rules Checked

| Rule ID | Category | Requirement |
|---------|----------|-------------|
| ACM-ABS-001 | Abstract | 100-250 words |
| ACM-KW-001 | Keywords | 3-10 keywords |
| ACM-CCS-001 | CCS Concepts | ACM CCS taxonomy concepts required |
| ACM-TTL-001 | Title | Maximum 12 words recommended |
| ACM-SEC-001 | Sections | Introduction, Related Work, Conclusion required |
| ACM-REF-001 | References | References section required |
| ACM-REF-002 | References | Consistent citation style |
| ACM-REF-003 | References | DOIs strongly recommended |
| ACM-AUTH-001 | Authors | ORCID iD recommended |
| ACM-AUTH-002 | Authors | Institutional affiliation required |
| ACM-ART-001 | Artifacts | Code/data availability statement encouraged |
| ACM-RIGHTS-001 | Rights | ACM rights block required in camera-ready |
| ACM-ETH-001 | Ethics | IRB statement for human subjects research |

#### ACM CCS Concepts Format

```
CCS Concepts: Security and privacy -> Intrusion detection systems;
              Computing methodologies -> Machine learning;
              Networks -> Network security.
```

Find your CCS concepts at: https://dl.acm.org/ccs

---

### IET

**Applies to:** IET Communications, IET Networks, IET Cyber-Systems and Security, IET Information Security, Electronics Letters, etc.

**Reference:** [IET Research Author Guide](https://ietresearch.onlinelibrary.wiley.com/hub/authors)

#### Rules Checked

| Rule ID | Category | Requirement |
|---------|----------|-------------|
| IET-ABS-001 | Abstract | 100-200 words |
| IET-ABS-002 | Abstract | No citations or references |
| IET-ABS-003 | Abstract | No mathematical expressions |
| IET-KW-001 | Keywords | 4-8 keywords |
| IET-SPELL-001 | Spelling | British English required |
| IET-REF-001 | References | References required |
| IET-REF-002 | References | Volume/issue/page details required |
| IET-SEC-001 | Sections | Introduction and Conclusion required |
| IET-NOT-001 | Notation | "equation (N)" not "Eq. (N)" |
| IET-NOT-002 | Notation | Check "Figure" vs "Fig." usage |
| IET-BIO-001 | Authors | Short author biography recommended |
| IET-FUND-001 | Funding | Acknowledgments section required |
| IET-DATA-001 | Data | Data Availability Statement encouraged |

#### British English Quick Reference (IET/BCS)

| American | British |
|----------|---------|
| analyze | analyse |
| optimize | optimise |
| recognize | recognise |
| utilization | utilisation |
| color | colour |
| center | centre |
| fiber | fibre |
| program (software) | programme (general); program (CS) |
| defense | defence |
| license (noun) | licence |

#### IET Reference Format

```
[1] Smith, J., Jones, B.: 'Title of the article', IET Commun., 2023, 17, (8), pp. 1234-1245.
    doi: 10.1049/cmu2.12345
```

---

### BCS

**Applies to:** The Computer Journal (Oxford University Press / BCS), BCS eWiC conference proceedings.

**Reference:** [The Computer Journal Author Instructions](https://academic.oup.com/comjnl/pages/General_Instructions)

#### Rules Checked

| Rule ID | Category | Requirement |
|---------|----------|-------------|
| BCS-ABS-001 | Abstract | 100-250 words |
| BCS-ABS-002 | Abstract | Avoid "This paper..." opening |
| BCS-ABS-003 | Abstract | No figure/table references |
| BCS-KW-001 | Keywords | 4-8 keywords |
| BCS-SPELL-001 | Spelling | British English required |
| BCS-TTL-001 | Title | Spell out acronyms unless universally known |
| BCS-TTL-002 | Title | No question marks in title |
| BCS-TTL-003 | Title | Maximum 15 words |
| BCS-SEC-001 | Sections | Introduction and Conclusion required |
| BCS-SEC-002 | Sections | Maximum 3 levels of nesting |
| BCS-REF-001 | References | References required |
| BCS-REF-002 | References | Numbered [N] style (not author-year) |
| BCS-REF-003 | References | Full journal names (not abbreviations) |
| BCS-FIG-001 | Figures | Sequential numbering |
| BCS-ACK-001 | Compliance | Acknowledgements (British spelling) required |
| BCS-COI-001 | Compliance | Conflict of Interest declaration required |
| BCS-LEN-001 | Length | Warning if > 10,000 words |
| BCS-LEN-002 | Length | Warning if < 2,000 words |

---

## 9. External Service Integrations

### Overview

AcademicGuard is fully functional without any external APIs. External services
provide enhanced coverage as optional additions.

| Service | Function | Pricing | API Key Required |
|---------|----------|---------|-----------------|
| CrossRef | Title/DOI deduplication | Free | No |
| LanguageTool | Grammar (offline) | Free | No |
| LanguageTool Premium | Enhanced grammar | Paid | Yes (URL) |
| ZeroGPT | Third-party AI detection | Free tier | Yes |
| GPTZero | Third-party AI detection | Free tier | Yes |
| Turnitin iThenticate | Institutional plagiarism | Institutional | Yes |
| Copyscape | Web plagiarism | Pay-per-search | Yes |

### Checking Service Status

```bash
academicguard services
```

Output example:
```
 Service              Status     Auth                  Note
 Turnitin iThenticate NOT SET    Bearer token          Requires institutional license
 Copyscape Premium    NOT SET    Username + API key    Pay-per-search pricing
 ZeroGPT AI Detector  ENABLED    API key               Free tier available
 CrossRef Metadata    ENABLED    None (free open API)  Always available
 LanguageTool (local) ENABLED    None                  Downloads on first use
```

---

## 10. Report Formats

### JSON Report

```bash
academicguard analyze paper.pdf --venue ieee --json report.json
```

Structure:
```json
{
  "document_title": "Paper Title",
  "venue": "IEEE",
  "timestamp": "2026-05-05T10:30:00+00:00",
  "overall_score": 0.823,
  "overall_label": "WARN",
  "modules": [
    {
      "module": "AI Detector",
      "score": 0.91,
      "label": "PASS",
      "summary": "LOW AI signal (9%) -- text appears predominantly human-written.",
      "metadata": {
        "overall_ai_probability": 0.09,
        "perplexity_score": 0.12,
        "burstiness_score": 0.78,
        "verdict": "LIKELY_HUMAN"
      },
      "findings": []
    },
    ...
  ]
}
```

### HTML Report

```bash
academicguard analyze paper.pdf --venue ieee --html report.html
```

Opens as a self-contained HTML file with:
- Overall score badge (PASS/WARN/FAIL)
- Per-module score bars
- Findings table with severity, message, location, and suggestion
- No external dependencies -- works offline

### Plain Text (console)

All analysis is printed to stdout by default with color-coded output.
Pipe to a file for plain text:

```bash
academicguard analyze paper.pdf --venue ieee 2>&1 | tee analysis.txt
```

---

## 11. Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TURNITIN_API_KEY` | Turnitin iThenticate API key | `ith_key_xxx...` |
| `COPYSCAPE_USER` | Copyscape username | `myusername` |
| `COPYSCAPE_KEY` | Copyscape API key | `key_xxx...` |
| `ZEROGPT_API_KEY` | ZeroGPT API key | `zgpt_xxx...` |
| `GPTZERO_API_KEY` | GPTZero API key | `gptz_xxx...` |
| `LANGUAGETOOL_URL` | LanguageTool server URL | `http://localhost:8010` |
| `TRANSFORMERS_CACHE` | HuggingFace model cache directory | `~/.cache/hf` |
| `ACADEMICGUARD_LOG` | Log level | `DEBUG`, `INFO`, `WARNING` |

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export TURNITIN_API_KEY="your_key_here"
export ZEROGPT_API_KEY="your_key_here"
```

---

## 12. Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | UTF-8 encoding |
| Microsoft Word | `.docx` | Requires `python-docx` |
| PDF | `.pdf` | Requires `pdfminer.six`; scanned PDFs not supported |
| LaTeX | `.tex`, `.latex` | Commands stripped; structure extracted from `\section{}` etc. |

**Scanned PDFs** are not supported. Run OCR first (e.g., `tesseract paper.pdf paper txt`).

**LaTeX tips:**
- Use `\begin{abstract}...\end{abstract}` for reliable abstract extraction
- Use `\keywords{...}` or `\begin{keywords}...\end{keywords}`
- Compile to PDF for best results

---

## 13. Extending AcademicGuard

### Adding a New Venue

Create a new style checker inheriting from `BaseStyleChecker`:

```python
# academicguard/style/springer.py
from academicguard.style.base import BaseStyleChecker
from academicguard.core.document import Document
from academicguard.core.report import Finding

class SpringerStyleChecker(BaseStyleChecker):
    venue_name = "Springer"
    venue_url = "https://www.springer.com/authors"

    def _check(self, doc: Document) -> list[Finding]:
        findings = []

        # Abstract: 150-250 words
        f = self._check_abstract_length(doc, 150, 250, "SPR-ABS-001")
        if f:
            findings.append(f)

        # Add your custom rules here
        return findings
```

Register in `academicguard/style/__init__.py`:
```python
from academicguard.style.springer import SpringerStyleChecker
VENUE_REGISTRY["springer"] = SpringerStyleChecker
```

### Adding a New Detector

Implement the analyze method returning a `ModuleResult`:

```python
from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult

class MyDetector:
    def analyze(self, doc: Document) -> ModuleResult:
        findings = []
        # ... your analysis logic ...
        score = 0.85
        return ModuleResult(
            module="My Detector",
            score=score,
            label="PASS" if score >= 0.80 else "WARN",
            summary="My analysis summary.",
            findings=findings,
            metadata={"custom_metric": 42},
        )
```

---

## 14. Troubleshooting

### GPT-2 model download fails
```bash
# Use heuristic mode (no download needed)
academicguard ai paper.pdf --no-transformer

# Or set cache directory to a location with more space
export TRANSFORMERS_CACHE="/path/with/space"
```

### LanguageTool fails to start
```bash
# Use Docker instead
docker run -d -p 8010:8010 silviof/docker-languagetool
export LANGUAGETOOL_URL="http://localhost:8010"

# Or skip grammar check
academicguard analyze paper.pdf --skip-grammar
```

### PDF parsing gives garbled text
- Ensure the PDF is not scanned (image-only): try opening in a PDF viewer
- Run OCR first: `ocrmypdf input.pdf output.pdf && academicguard analyze output.pdf`
- Try converting to DOCX via LibreOffice: `soffice --convert-to docx paper.pdf`

### False positives in AI detection
- Short texts (< 200 words) are unreliable -- use longer excerpts
- Highly technical text with fixed terminology may score higher
- Use `--no-transformer` for a pure heuristic score as comparison
- Combine with ZeroGPT or GPTZero for independent validation

### Installation issues on Apple Silicon (M1/M2/M3)
```bash
# Ensure arm64 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install academicguard
```

---

## 15. Industry Tool Comparison

AcademicGuard is designed to complement -- not replace -- commercial tools.

| Feature | AcademicGuard | Turnitin | iThenticate | Grammarly | GPTZero |
|---------|:------------:|:--------:|:-----------:|:---------:|:-------:|
| AI Detection | Yes (local) | No | No | No | Yes |
| Plagiarism (web) | Via Copyscape* | Yes | Yes | No | No |
| Plagiarism (local corpus) | Yes | No | No | No | No |
| Grammar Check | Yes | No | No | Yes | No |
| IEEE Style Check | Yes | No | No | No | No |
| Elsevier Style Check | Yes | No | No | No | No |
| ACM Style Check | Yes | No | No | No | No |
| IET Style Check | Yes | No | No | No | No |
| BCS Style Check | Yes | No | No | No | No |
| Works offline | Yes | No | No | No | No |
| Open source | Yes | No | No | No | No |
| Free | Yes | No | No | Partial | Partial |
| API | Yes | Yes | Yes | Enterprise | Yes |

*Requires Copyscape API key.

**Recommended workflow:**
1. Run AcademicGuard locally for immediate feedback during writing
2. Run Turnitin/iThenticate for final submission plagiarism check (institutional)
3. Use ZeroGPT or GPTZero as secondary AI-detection validation

---

## 16. Venue Compliance Checklists

### IEEE Submission Checklist

- [ ] Abstract: 150-250 words, no citations
- [ ] Index Terms: 3-10, lowercase, from IEEE Thesaurus
- [ ] Title: Title case, no abbreviations unexplained
- [ ] Sections: Introduction, Related Work, Methodology, Results, Conclusion
- [ ] Citations: Numeric [N] style, ordered by first appearance
- [ ] Figures: "Fig. N" in body; self-explanatory captions
- [ ] Units: Space between number and SI unit (e.g., "5 GHz")
- [ ] "et al." correctly formatted with period
- [ ] Acronyms defined on first use; not used if only once
- [ ] No contractions or colloquial language
- [ ] AI detection: score < 25% AI probability recommended

### Elsevier Submission Checklist

- [ ] Highlights: 3-5 bullets, each <= 85 characters
- [ ] Abstract: 150-300 words, structured if required by journal
- [ ] Keywords: 4-8, no trailing punctuation
- [ ] Declaration of Competing Interests present
- [ ] CRediT Author Contribution Statement present
- [ ] Funding and Ethics statements present
- [ ] Data Availability Statement present
- [ ] DOIs for all references where available
- [ ] IMRAD structure (Introduction, Methods, Results, Discussion, Conclusion)

### ACM Submission Checklist

- [ ] CCS Concepts: from ACM taxonomy (https://dl.acm.org/ccs)
- [ ] Abstract: 100-250 words
- [ ] ORCID iDs for all authors
- [ ] Author affiliations: institution, city, country, email
- [ ] ACM rights block (camera-ready only)
- [ ] Artifact availability statement (for reproducibility badges)
- [ ] IRB/ethics statement for human subjects research
- [ ] DOIs for references

### IET Submission Checklist

- [ ] Abstract: 100-200 words, no citations or equations
- [ ] Keywords: 4-8
- [ ] British English spelling throughout
- [ ] Author biographies (50-100 words each)
- [ ] Acknowledgments section with funding disclosure
- [ ] Data Availability Statement
- [ ] "equation (N)" not "Eq. (N)"
- [ ] Reference format: Author(s): 'Title'. Journal, Year, Vol(Issue), pp.

### BCS / The Computer Journal Checklist

- [ ] Title: no question marks, no unexplained acronyms, <= 15 words
- [ ] Abstract: 100-250 words, no figures/tables/citations referenced
- [ ] Abstract does not begin with "This paper..."
- [ ] Keywords: 4-8
- [ ] British English spelling throughout
- [ ] Acknowledgements (British spelling) present
- [ ] Conflict of Interest declaration present
- [ ] Numbered [N] references with full journal names
- [ ] Section hierarchy <= 3 levels
- [ ] Figures numbered sequentially
- [ ] Word count: 4000-8000 words (full article)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/academicguard/academicguard.git
cd academicguard
pip install -e ".[dev]"
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use AcademicGuard in your research, please cite:

```bibtex
@software{academicguard2026,
  title   = {AcademicGuard: Open-Source Academic Writing Integrity Toolkit},
  year    = {2026},
  url     = {https://github.com/academicguard/academicguard},
  version = {1.0.0},
  license = {MIT}
}
```

<div align="center">

# AcademicGuard

**Catch AI-flagged phrasing, plagiarism, grammar slips, and venue formatting mistakes**
**before your reviewer does, entirely on your own machine.**

[![Latest release](https://img.shields.io/github/v/release/sunilgentyala/academicguard?label=release)](https://github.com/sunilgentyala/academicguard/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![No API keys required](https://img.shields.io/badge/API%20keys-none%20required-brightgreen.svg)](HOWTO.md#9-local-detection-components)
[![Venues](https://img.shields.io/badge/venues-IEEE%20%7C%20Elsevier%20%7C%20ACM%20%7C%20IET%20%7C%20BCS-informational.svg)](HOWTO.md#8-venue-style-checkers)

[Quick Start](#quick-start) · [Full Documentation](HOWTO.md) · [Project Site](https://sunilgentyala.github.io/academicguard/) · [Changelog](#changelog) · [Contributing](CONTRIBUTING.md)

</div>

---

## What it does

AcademicGuard runs four checks against a paper -- `.txt`, `.docx`, `.pdf`, or
`.tex` -- and prints a scored, actionable report for each:

| Check | How | Details |
|---|---|---|
| **AI-content signal** | 9-signal local ensemble (GLTR, perplexity, burstiness, Zipf, Yule's K, hapax rate, n-gram entropy, stylometrics, semantic coherence) | [Section 5](HOWTO.md#5-ai-content-detection) |
| **Plagiarism / self-plagiarism** | Winnowing (MOSS-style) + MinHash/LSH + TF-IDF sentence similarity against your own corpus, plus a free CrossRef title lookup | [Section 6](HOWTO.md#6-plagiarism-detection) |
| **Grammar & academic register** | LanguageTool + contraction/colloquialism/wordiness/hedging rules + Flesch-Kincaid readability | [Section 7](HOWTO.md#7-grammar-and-academic-english) |
| **Venue formatting** | Rule-based checkers for **IEEE**, **Elsevier**, **ACM**, **IET**, and **BCS** -- abstract/keyword length, citation style, required sections, British vs. American spelling, and more | [Section 8](HOWTO.md#8-venue-style-checkers) |

Every check above runs **locally, with no API key and no account** -- the only
network call anywhere in the tool is an optional, keyless CrossRef title
lookup. See [what's local vs. what AcademicGuard deliberately doesn't
integrate](HOWTO.md#9-local-detection-components) (Turnitin, Copyscape,
ZeroGPT, GPTZero -- run those separately for a final institutional check).

## Quick Start

```bash
pip install academicguard
python -m spacy download en_core_web_sm

academicguard analyze paper.pdf --venue ieee --html report.html
```

```
AcademicGuard v1.0.1
File: paper.pdf  |  Venue: IEEE

┌──────────────┬────────────────────┬────────┬─────────────────────────────┐
│ Module       │ Score              │ Status │ Summary                     │
├──────────────┼────────────────────┼────────┼─────────────────────────────┤
│ AI Detector  │ ████████░░ 84%     │ PASS   │ AI probability 9% (high)... │
│ Plagiarism   │ █████████░ 92%     │ PASS   │ Low similarity (3%)...      │
│ Grammar      │ ███████░░░ 71%     │ WARN   │ 4 errors, 11 warnings...    │
│ IEEE Style   │ ████████░░ 85%     │ PASS   │ IEEE style: 0 errors...     │
└──────────────┴────────────────────┴────────┴─────────────────────────────┘

Overall: WARN (83%)
```

```python
import academicguard.api as ag

report = ag.analyze("paper.pdf", venue="ieee")
print(f"{report.overall_label} ({report.overall_score:.0%})")
report.save_html("report.html")
```

Exit codes are CI-friendly: `0` = PASS, `1` = WARN, `2` = FAIL. See the
[full CLI and Python API reference](HOWTO.md#3-cli-reference) for every flag
and module.

## Why AcademicGuard

- **Private by default.** Your unpublished paper never leaves your machine.
- **Multi-venue.** One tool for IEEE, Elsevier, ACM, IET, and BCS formatting rules, instead of five checklists.
- **Actionable, not just a score.** Every finding includes a rule ID, location, and a concrete suggestion.
- **CI-friendly.** Standard exit codes and a JSON report make it easy to gate a submission pipeline.
- **Cheap to run.** Skip the GPT-2 download entirely with `--no-transformer` for a heuristic-only pass in milliseconds.

## Documentation

- [HOWTO.md](HOWTO.md) -- complete CLI reference, Python API, per-venue rule tables, and troubleshooting
- [Project site](https://sunilgentyala.github.io/academicguard/) -- feature tour
- [CONTRIBUTING.md](CONTRIBUTING.md) -- how to add a venue checker or detection signal

## Changelog

Full release notes: [GitHub Releases](https://github.com/sunilgentyala/academicguard/releases).

### [v1.0.1](https://github.com/sunilgentyala/academicguard/releases/tag/v1.0.1) -- 2026-08-09

Bug-fix release; no breaking API changes.

- **Fixed:** plain-text abstract/keyword parsing for the common inline
  `Abstract-- text...` convention, which previously fell through to the
  title heuristic and left the abstract and keywords empty.
- **Fixed:** `TFIDFSimilarityChecker` scoring short-but-identical sentences
  as 0% similar.
- **Fixed:** `api.analyze_text()` silently reading a file from disk when the
  input text happened to match a real filename (e.g. `"LICENSE"`).
- **Fixed:** IET/BCS British-spelling checker flagging correctly-spelled
  British text (`"behaviour"`) and the noun `"practice"` as errors.
- **Fixed:** the main `analyze` pipeline now runs grammar checking in the
  right English variant (en-GB) for IET/BCS instead of always en-US.
- **Fixed:** HTML report footer no longer hardcodes the version string.
- **Removed:** four unused dependencies (`pydantic`, `numpy`, `pylatexenc`,
  `click`) and two phantom optional extras with no corresponding code.
- **Docs:** corrected `HOWTO.md` sections that described Turnitin/
  iThenticate, Copyscape, ZeroGPT, and GPTZero integrations and API-key
  environment variables that don't exist in the code; added the
  previously-missing `CONTRIBUTING.md`; new project site at
  [sunilgentyala.github.io/academicguard](https://sunilgentyala.github.io/academicguard/).

### [v1.0.0](https://github.com/sunilgentyala/academicguard/releases/tag/v1.0.0) -- 2026-05-06

Initial release -- AI content detection, local + CrossRef plagiarism
checking, grammar/register checking, and IEEE/Elsevier/ACM/IET/BCS venue
style checkers, with a CLI and Python API.

## License

MIT -- see [LICENSE](LICENSE).

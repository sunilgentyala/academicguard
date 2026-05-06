
# AcademicGuard

**Open-source academic writing integrity toolkit**

AI detection | Plagiarism | Grammar | IEEE | Elsevier | ACM | IET | BCS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **AI Content Detection** -- local GPT-2 perplexity + burstiness + phrase analysis; no external API required
- **Plagiarism Detection** -- MinHash/LSH local corpus + CrossRef + optional Turnitin/Copyscape
- **Grammar & Academic Register** -- LanguageTool + contraction/colloquial/wordiness checks
- **Venue Style Enforcement** -- IEEE, Elsevier, ACM, IET, and BCS rule-based checkers
- **HTML + JSON reports** -- shareable, self-contained output
- **CLI + Python API** -- integrate into CI/CD or use interactively

## Install

```bash
pip install academicguard
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
academicguard analyze paper.pdf --venue ieee --html report.html
```

See the full [HOWTO.md](HOWTO.md) for complete documentation.

## License

MIT

# Contributing to AcademicGuard

Thanks for considering a contribution. AcademicGuard is a small, focused
toolkit and PRs of any size are welcome -- new venue checkers, additional
detection signals, bug fixes, or documentation cleanups.

## Getting set up

```bash
git clone https://github.com/sunilgentyala/academicguard.git
cd academicguard
pip install -e ".[dev]"
python -m spacy download en_core_web_sm   # optional, needed for spaCy-based grammar checks
pytest tests/
```

The core parsing and scoring logic (`academicguard/core`, `academicguard/style`)
has no heavy dependencies, so `pytest tests/` runs in well under a second even
without `torch`/`transformers`/`spacy` installed -- those are only imported
lazily, inside the detectors that need them, and every call site falls back
to a pure-Python heuristic if the optional dependency is missing.

## Before opening a PR

- Run `pytest tests/` and add tests for new behavior.
- Run `ruff check .` and `black .` if you have them installed (`pip install -e ".[dev]"` gets you both).
- Keep new venue/style checkers consistent with the existing pattern in
  `academicguard/style/base.py` -- see the "Extending AcademicGuard" section
  of [HOWTO.md](HOWTO.md) for a worked example.
- If you're adding a new dependency, make sure it's actually imported
  somewhere; the project intentionally keeps its dependency list tight.

## Reporting bugs

Open an issue at https://github.com/sunilgentyala/academicguard/issues with:
- The command or API call you ran
- What you expected vs. what happened
- A minimal text/docx/pdf/tex sample that reproduces it, if possible

## Adding a new publication venue

See [Section 13 of HOWTO.md](HOWTO.md#13-extending-academicguard) for a
step-by-step example (subclass `BaseStyleChecker`, register it in
`academicguard/style/__init__.py`, add tests).

## Code of conduct

Be respectful and constructive. This is a small open-source project run in
contributors' spare time.

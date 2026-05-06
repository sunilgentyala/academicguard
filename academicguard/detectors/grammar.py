"""Grammar and academic English checker.

Layers:
1. LanguageTool (local server or hosted API) -- comprehensive grammar/spelling
2. Academic register rules -- passive voice, hedging, informal language
3. Readability metrics -- Flesch-Kincaid for academic writing
4. Sentence structure analysis via spaCy
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult


# ------------------------------------------------------------------ #
# LanguageTool wrapper
# ------------------------------------------------------------------ #

class LanguageToolChecker:
    """Wraps language-tool-python for grammar/spelling checks."""

    def __init__(self, language: str = "en-US", remote_url: Optional[str] = None):
        self._tool = None
        self.language = language
        self.remote_url = remote_url

    def _get_tool(self):
        if self._tool is not None:
            return self._tool
        try:
            import language_tool_python
            if self.remote_url:
                self._tool = language_tool_python.LanguageToolPublicAPI(
                    self.language, host=self.remote_url
                )
            else:
                self._tool = language_tool_python.LanguageTool(self.language)
        except Exception:
            self._tool = None
        return self._tool

    def check(self, text: str) -> list[dict]:
        tool = self._get_tool()
        if tool is None:
            return []
        try:
            matches = tool.check(text)
            return [
                {
                    "message": m.message,
                    "offset": m.offset,
                    "length": m.errorLength,
                    "replacements": m.replacements[:3],
                    "rule_id": m.ruleId,
                    "category": m.category,
                    "context": text[max(0, m.offset-30):m.offset+m.errorLength+30],
                }
                for m in matches
            ]
        except Exception:
            return []

    def close(self):
        if self._tool:
            try:
                self._tool.close()
            except Exception:
                pass


# ------------------------------------------------------------------ #
# Academic register rules
# ------------------------------------------------------------------ #

# Informal contractions
_CONTRACTIONS = re.compile(
    r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't|isn't|aren't|"
    r"wasn't|weren't|hasn't|haven't|hadn't|it's|that's|there's|they're|"
    r"we're|you're|I'm|I've|I'd|I'll|we've|we'd|we'll|they've|they'd|they'll)\b",
    re.I,
)

# Colloquial words
_COLLOQUIAL = re.compile(
    r"\b(a lot|lots of|tons of|kind of|sort of|a bit|pretty (good|bad|much|well)|"
    r"basically|actually|literally|really|very very|stuff|things|get|got|big deal|"
    r"figure out|come up with|look into|go ahead|end up|turn out)\b",
    re.I,
)

# First-person singular (context-dependent; flag but don't error)
_FIRST_PERSON_SINGULAR = re.compile(r"\b(I|me|my|myself|mine)\b")

# Vague intensifiers
_VAGUE_INTENSIFIERS = re.compile(
    r"\b(very|extremely|highly|incredibly|absolutely|totally|completely|"
    r"entirely|utterly|awfully|terribly|quite|rather|fairly|somewhat)\s+"
    r"(good|bad|large|small|fast|slow|high|low|important|significant)\b",
    re.I,
)

# Hedging phrases (good in academic writing -- check for absence)
_HEDGING = re.compile(
    r"\b(may|might|could|possibly|probably|perhaps|approximately|seems? to|"
    r"appear[s]? to|suggest[s]?|indicate[s]?|imply|implies)\b",
    re.I,
)

# Wordiness patterns
_WORDY_PHRASES = {
    r"\bin order to\b": "to",
    r"\bdue to the fact that\b": "because",
    r"\bat this point in time\b": "now",
    r"\bfor the purpose of\b": "to",
    r"\bhas the ability to\b": "can",
    r"\bis able to\b": "can",
    r"\bin the event that\b": "if",
    r"\bwith regard to\b": "regarding",
    r"\bwith respect to\b": "regarding",
    r"\bprior to\b": "before",
    r"\bsubsequent to\b": "after",
    r"\bsufficient number of\b": "enough",
    r"\bthe majority of\b": "most",
}

# Passive voice simple detector
_PASSIVE_BE = re.compile(r"\b(is|are|was|were|been|being|be)\s+\w+ed\b", re.I)


# ------------------------------------------------------------------ #
# Readability
# ------------------------------------------------------------------ #

def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level -- target 12-16 for academic papers."""
    words = re.findall(r"\b\w+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]+", text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / len(sentences)      # avg sentence length
    asw = syllables / len(words)           # avg syllables per word
    return 0.39 * asl + 11.8 * asw - 15.59


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease -- academic papers typically 30-50."""
    words = re.findall(r"\b\w+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]+", text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ------------------------------------------------------------------ #
# spaCy sentence analysis
# ------------------------------------------------------------------ #

def analyze_sentences_spacy(text: str) -> dict:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:50000])  # limit for performance
        sent_lengths = [len(list(sent)) for sent in doc.sents]
        passive_count = sum(
            1 for token in doc if token.dep_ == "nsubjpass"
        )
        return {
            "avg_sentence_length": statistics.mean(sent_lengths) if sent_lengths else 0,
            "sentence_length_std": statistics.pstdev(sent_lengths) if len(sent_lengths) > 1 else 0,
            "passive_count": passive_count,
            "sentence_count": len(sent_lengths),
        }
    except Exception:
        return {}


# ------------------------------------------------------------------ #
# Main grammar checker
# ------------------------------------------------------------------ #

class GrammarChecker:
    """Comprehensive grammar and academic English style checker."""

    def __init__(self, language: str = "en-US", lt_remote_url: Optional[str] = None):
        self._lt = LanguageToolChecker(language, lt_remote_url)
        self.language = language

    def analyze(self, doc: Document) -> ModuleResult:
        text = doc.raw_text
        findings: list[Finding] = []
        error_count = 0
        warning_count = 0

        # 1. LanguageTool grammar/spelling
        lt_matches = self._lt.check(text)
        for m in lt_matches[:50]:  # cap at 50 to avoid noise
            sev = "error" if m["category"] in {"TYPOS", "GRAMMAR"} else "warning"
            if sev == "error":
                error_count += 1
            else:
                warning_count += 1
            suggestion = m["replacements"][0] if m["replacements"] else ""
            findings.append(Finding(
                category="grammar",
                severity=sev,
                message=m["message"],
                suggestion=f'Try: "{suggestion}"' if suggestion else "",
                rule_id=f"LT-{m['rule_id']}",
                context=m["context"],
            ))

        # 2. Academic register checks
        findings.extend(self._check_register(text))

        # 3. Readability
        fk_grade = flesch_kincaid_grade(text)
        fk_ease = flesch_reading_ease(text)
        if fk_grade < 10:
            findings.append(Finding(
                category="grammar",
                severity="warning",
                message=f"Flesch-Kincaid grade {fk_grade:.1f} is below academic range (12-16).",
                suggestion="Use more domain-specific vocabulary and complex sentence structures.",
                rule_id="READ-001",
            ))
        if fk_ease > 60:
            findings.append(Finding(
                category="grammar",
                severity="info",
                message=f"Flesch reading ease {fk_ease:.1f} is high -- text may be too informal.",
                suggestion="Academic papers typically score 30-50.",
                rule_id="READ-002",
            ))

        # 4. Sentence structure via spaCy
        spacy_info = analyze_sentences_spacy(text[:30000])
        if spacy_info:
            avg_len = spacy_info.get("avg_sentence_length", 0)
            if avg_len > 40:
                findings.append(Finding(
                    category="grammar",
                    severity="warning",
                    message=f"Average sentence length is {avg_len:.0f} words -- consider shorter sentences.",
                    suggestion="Break long sentences into 20-30 word units for clarity.",
                    rule_id="SENT-001",
                ))
            if avg_len < 10 and spacy_info.get("sentence_count", 0) > 5:
                findings.append(Finding(
                    category="grammar",
                    severity="info",
                    message=f"Average sentence length is {avg_len:.0f} words -- may read as choppy.",
                    suggestion="Combine related short sentences for better academic flow.",
                    rule_id="SENT-002",
                ))

        # Score: errors weight 2x warnings
        total_issues = error_count * 2 + warning_count
        words = max(1, len(text.split()))
        issue_rate = total_issues / (words / 100)
        score = max(0.0, 1.0 - (issue_rate / 15.0))
        label = "PASS" if score >= 0.80 else ("WARN" if score >= 0.55 else "FAIL")

        return ModuleResult(
            module="Grammar",
            score=score,
            label=label,
            summary=self._summary(error_count, warning_count, fk_grade, fk_ease),
            findings=findings,
            metadata={
                "lt_errors": error_count,
                "lt_warnings": warning_count,
                "fk_grade_level": round(fk_grade, 1),
                "flesch_reading_ease": round(fk_ease, 1),
                "avg_sentence_length": round(spacy_info.get("avg_sentence_length", 0), 1),
                "passive_constructions": spacy_info.get("passive_count", "n/a"),
            },
        )

    def _check_register(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        # Contractions
        for m in _CONTRACTIONS.finditer(text):
            findings.append(Finding(
                category="grammar",
                severity="error",
                message=f'Contraction "{m.group()}" is inappropriate in formal academic writing.',
                location=f"offset {m.start()}",
                suggestion=f'Expand to full form (e.g., "do not", "cannot").',
                rule_id="REG-001",
                context=text[max(0, m.start()-20):m.end()+20],
            ))

        # Colloquial language
        for m in _COLLOQUIAL.finditer(text):
            findings.append(Finding(
                category="grammar",
                severity="warning",
                message=f'Colloquial expression "{m.group()}" weakens academic register.',
                location=f"offset {m.start()}",
                suggestion="Replace with formal equivalents.",
                rule_id="REG-002",
                context=text[max(0, m.start()-20):m.end()+20],
            ))

        # First-person singular (warn, not error -- some venues allow it)
        singular_matches = list(_FIRST_PERSON_SINGULAR.finditer(text))
        if len(singular_matches) > 5:
            findings.append(Finding(
                category="grammar",
                severity="info",
                message=f'First-person singular ("I") used {len(singular_matches)} times. '
                        f'Many venues prefer "we" or passive constructions.',
                suggestion='Use "we" for multi-author papers; check venue guidelines.',
                rule_id="REG-003",
            ))

        # Vague intensifiers
        for m in _VAGUE_INTENSIFIERS.finditer(text):
            findings.append(Finding(
                category="grammar",
                severity="info",
                message=f'Vague intensifier: "{m.group()}" -- prefer quantified claims.',
                location=f"offset {m.start()}",
                suggestion="Quantify with numerical values instead.",
                rule_id="REG-004",
                context=text[max(0, m.start()-20):m.end()+20],
            ))

        # Wordiness
        for pattern, replacement in _WORDY_PHRASES.items():
            for m in re.finditer(pattern, text, re.I):
                findings.append(Finding(
                    category="grammar",
                    severity="info",
                    message=f'Wordy phrase: "{m.group()}"',
                    location=f"offset {m.start()}",
                    suggestion=f'Consider: "{replacement}"',
                    rule_id="REG-005",
                    context=text[max(0, m.start()-20):m.end()+20],
                ))

        # Check hedging presence in abstracts (absence is a problem)
        abstract_text = text[:600]  # rough abstract approximation
        if len(abstract_text) > 100 and not _HEDGING.search(abstract_text):
            findings.append(Finding(
                category="grammar",
                severity="info",
                message="No hedging language detected in the abstract.",
                suggestion='Add hedging expressions (e.g., "The results suggest...", "This approach may...") for appropriate epistemic caution.',
                rule_id="REG-006",
            ))

        return findings

    def _summary(self, errors: int, warnings: int, fk_grade: float, fk_ease: float) -> str:
        parts = [f"{errors} grammar errors, {warnings} style warnings"]
        parts.append(f"FK grade {fk_grade:.1f}")
        parts.append(f"reading ease {fk_ease:.0f}")
        return "; ".join(parts) + "."

    def close(self):
        self._lt.close()

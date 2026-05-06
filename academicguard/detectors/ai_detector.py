"""AI content detector using perplexity scoring and burstiness analysis.

Detection methodology:
- Perplexity (GPT-2): AI text has lower perplexity -- it's more predictable.
- Burstiness: Human writing has high variance in sentence length; AI writing is uniform.
- Repetition density: AI tends to repeat phrases and transitional patterns.
- Vocabulary richness: type-token ratio and hapax legomena rate.

Interpretation (ensemble):
  score >= 0.75  -> likely AI-generated
  0.45 <= score < 0.75 -> uncertain / mixed
  score < 0.45   -> likely human-written
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult


@dataclass
class AIDetectionResult:
    overall_probability: float          # 0.0 = human, 1.0 = AI
    perplexity_score: float
    burstiness_score: float
    repetition_score: float
    vocabulary_score: float
    confidence: str                      # "high", "medium", "low"
    label: str                           # "LIKELY_AI", "UNCERTAIN", "LIKELY_HUMAN"
    per_paragraph: list[dict]


class AIDetector:
    """Ensemble AI content detector -- no external API required."""

    # Common AI "glue" phrases
    _AI_PHRASES = [
        r"\bin conclusion\b", r"\bto summarize\b", r"\bfurthermore\b",
        r"\bmoreover\b", r"\bit is worth noting\b", r"\bit is important to note\b",
        r"\bit should be noted\b", r"\bthis paper (presents|proposes|investigates)\b",
        r"\bthe results (show|demonstrate|indicate|suggest)\b",
        r"\bsignificant(ly)?\b.*\b(improvement|reduction|increase)\b",
        r"\bstate[- ]of[- ]the[- ]art\b", r"\bnovel (approach|method|framework)\b",
        r"\brobust(ness)?\b", r"\bseamlessly\b", r"\bcomprehensive\b",
        r"\bleverag(e|ing)\b", r"\bdelve\b", r"\btestimonial\b",
    ]

    def __init__(self, use_transformer: bool = True, model_name: str = "gpt2"):
        self._model = None
        self._tokenizer = None
        self._use_transformer = use_transformer
        self._model_name = model_name

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()
        except ImportError:
            self._use_transformer = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, doc: Document) -> ModuleResult:
        text = (doc.abstract + "\n\n" + doc.body_text).strip() or doc.raw_text
        result = self._detect(text)
        findings = self._build_findings(result, doc)
        score = 1.0 - result.overall_probability
        label = "PASS" if score >= 0.75 else ("WARN" if score >= 0.45 else "FAIL")
        return ModuleResult(
            module="AI Detector",
            score=score,
            label=label,
            summary=self._summary(result),
            findings=findings,
            metadata={
                "overall_ai_probability": round(result.overall_probability, 3),
                "perplexity_score": round(result.perplexity_score, 3),
                "burstiness_score": round(result.burstiness_score, 3),
                "repetition_score": round(result.repetition_score, 3),
                "vocabulary_richness": round(result.vocabulary_score, 3),
                "confidence": result.confidence,
                "verdict": result.label,
            },
        )

    def analyze_text(self, text: str) -> AIDetectionResult:
        return self._detect(text)

    # ------------------------------------------------------------------ #
    # Detection pipeline
    # ------------------------------------------------------------------ #

    def _detect(self, text: str) -> AIDetectionResult:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 50]

        perp = self._perplexity_signal(text)
        burst = self._burstiness_signal(text)
        rep = self._repetition_signal(text)
        vocab = self._vocabulary_signal(text)

        # Paragraph-level signals
        per_para = []
        for i, para in enumerate(paragraphs[:20]):
            pp = self._perplexity_signal(para)
            pb = self._burstiness_signal(para)
            pr = self._repetition_signal(para)
            p_prob = self._ensemble(pp, pb, pr, vocab)
            per_para.append({
                "paragraph": i + 1,
                "snippet": para[:120] + ("..." if len(para) > 120 else ""),
                "ai_probability": round(p_prob, 3),
            })

        overall_prob = self._ensemble(perp, burst, rep, vocab)

        if overall_prob > 0.75:
            label = "LIKELY_AI"
            confidence = "high" if overall_prob > 0.88 else "medium"
        elif overall_prob > 0.45:
            label = "UNCERTAIN"
            confidence = "low"
        else:
            label = "LIKELY_HUMAN"
            confidence = "high" if overall_prob < 0.25 else "medium"

        return AIDetectionResult(
            overall_probability=overall_prob,
            perplexity_score=perp,
            burstiness_score=burst,
            repetition_score=rep,
            vocabulary_score=vocab,
            confidence=confidence,
            label=label,
            per_paragraph=per_para,
        )

    def _ensemble(self, perp: float, burst: float, rep: float, vocab: float) -> float:
        """Weighted ensemble of individual signals -> probability of AI."""
        return min(1.0, max(0.0,
            0.35 * perp + 0.25 * (1.0 - burst) + 0.25 * rep + 0.15 * (1.0 - vocab)
        ))

    # ------------------------------------------------------------------ #
    # Individual signals
    # ------------------------------------------------------------------ #

    def _perplexity_signal(self, text: str) -> float:
        """
        Returns probability of AI based on perplexity.
        Low perplexity -> model found text very predictable -> likely AI.
        Falls back to heuristic n-gram model when transformers unavailable.
        """
        if self._use_transformer:
            self._load_model()
        if self._model is not None:
            return self._transformer_perplexity(text)
        return self._heuristic_perplexity(text)

    def _transformer_perplexity(self, text: str) -> float:
        import torch
        tokens = self._tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(tokens, labels=tokens)
            loss = outputs.loss.item()
        ppl = math.exp(loss)
        # GPT-2 perplexity: human ~200-800, AI ~30-120
        # Normalize to 0-1 probability of AI
        normalized = max(0.0, min(1.0, 1.0 - (ppl - 20) / 700))
        return normalized

    def _heuristic_perplexity(self, text: str) -> float:
        """
        Heuristic fallback: measure character-level bigram entropy.
        Low entropy (more predictable) -> higher AI probability.
        """
        if len(text) < 20:
            return 0.5
        bigrams: dict[str, int] = {}
        chars = text.lower()
        for i in range(len(chars) - 1):
            bg = chars[i:i+2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
        total = sum(bigrams.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in bigrams.values() if c > 0)
        # English bigram entropy ~9-11 bits; AI text tends to sit slightly lower
        normalized = max(0.0, min(1.0, (11.0 - entropy) / 4.0))
        return normalized

    def _burstiness_signal(self, text: str) -> float:
        """
        Burstiness: coefficient of variation of sentence lengths.
        High CV -> bursty (human). Low CV -> uniform (AI).
        Returns human-burstiness [0,1]; caller inverts it for AI probability.
        """
        sentences = re.findall(r"[^.!?]+[.!?]+", text)
        lengths = [len(s.split()) for s in sentences if len(s.split()) >= 2]
        if len(lengths) < 4:
            return 0.5
        mean_l = statistics.mean(lengths)
        std_l = statistics.pstdev(lengths)
        cv = std_l / mean_l if mean_l > 0 else 0
        # CV > 0.6 -> very bursty (human). CV < 0.25 -> uniform (AI).
        normalized = min(1.0, cv / 0.65)
        return normalized

    def _repetition_signal(self, text: str) -> float:
        """
        Repetition density: frequency of known AI filler phrases.
        Higher repetition -> higher AI probability.
        """
        text_lower = text.lower()
        word_count = max(1, len(text.split()))
        hits = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self._AI_PHRASES
        )
        density = hits / (word_count / 100)
        return min(1.0, density / 5.0)

    def _vocabulary_signal(self, text: str) -> float:
        """
        Type-token ratio + hapax legomena rate.
        Higher TTR -> richer vocabulary (more human-like).
        Returns richness [0,1]; caller inverts for AI probability.
        """
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        if len(words) < 20:
            return 0.5
        unique = set(words)
        ttr = len(unique) / len(words)
        hapax = sum(1 for w in unique if words.count(w) == 1)
        hapax_rate = hapax / len(unique)
        richness = (ttr * 0.6 + hapax_rate * 0.4)
        return min(1.0, richness * 1.5)

    # ------------------------------------------------------------------ #
    # Finding generation
    # ------------------------------------------------------------------ #

    def _build_findings(self, result: AIDetectionResult, doc: Document) -> list[Finding]:
        findings: list[Finding] = []

        if result.label == "LIKELY_AI":
            findings.append(Finding(
                category="ai",
                severity="error",
                message=f"Text shows strong AI-generation signals ({result.overall_probability:.0%} AI probability).",
                suggestion="Revise substantially in your own voice; rephrase or remove AI-typical phrasing.",
                rule_id="AI-001",
            ))
        elif result.label == "UNCERTAIN":
            findings.append(Finding(
                category="ai",
                severity="warning",
                message=f"Mixed signals -- {result.overall_probability:.0%} AI probability. Sections may be AI-assisted.",
                suggestion="Review AI-flagged paragraphs and ensure they reflect original thought.",
                rule_id="AI-002",
            ))

        # Flag high-probability paragraphs
        for para in result.per_paragraph:
            if para["ai_probability"] >= 0.75:
                findings.append(Finding(
                    category="ai",
                    severity="warning",
                    message=f"Paragraph {para['paragraph']} shows {para['ai_probability']:.0%} AI probability.",
                    location=f"Paragraph {para['paragraph']}",
                    context=para["snippet"],
                    suggestion="Rewrite in your own voice with specific technical detail.",
                    rule_id="AI-003",
                ))

        # Check for known AI phrases in document text
        text_lower = doc.raw_text.lower()
        flagged_phrases = []
        for pattern in self._AI_PHRASES:
            for m in re.finditer(pattern, text_lower):
                phrase = doc.raw_text[m.start():m.end()]
                if phrase not in flagged_phrases:
                    flagged_phrases.append(phrase)
                    findings.append(Finding(
                        category="ai",
                        severity="info",
                        message=f'Common AI-generated phrase detected: "{phrase}"',
                        suggestion="Replace with more specific, domain-precise language.",
                        rule_id="AI-004",
                        context=phrase,
                    ))
                if len(flagged_phrases) >= 10:
                    break
            if len(flagged_phrases) >= 10:
                break

        return findings

    def _summary(self, result: AIDetectionResult) -> str:
        pct = f"{result.overall_probability:.0%}"
        if result.label == "LIKELY_AI":
            return f"HIGH AI signal ({pct}) -- text is likely AI-generated. Perplexity={result.perplexity_score:.2f}, burstiness={result.burstiness_score:.2f}."
        if result.label == "UNCERTAIN":
            return f"UNCERTAIN ({pct} AI probability) -- mixed human/AI signals detected."
        return f"LOW AI signal ({pct}) -- text appears predominantly human-written."

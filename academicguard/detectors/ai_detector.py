"""
AcademicGuard -- Self-contained AI Content Detector
====================================================
Zero external API calls. All algorithms implemented locally.

Detection pipeline (9 signals, ensemble-weighted):

  Signal 1  -- GLTR Token Rank Analysis (MIT-IBM Watson AI Lab method)
               Rank each word in GPT-2's probability distribution.
               AI text concentrates in top-10 (green zone); human text is spread.

  Signal 2  -- Perplexity Sliding Window
               Low perplexity per window = model found text highly predictable = AI.

  Signal 3  -- Burstiness Coefficient
               Human writing has high variance in sentence length (bursty).
               AI writing is uniformly structured (low CV).

  Signal 4  -- Zipf's Law Deviation
               Human vocabulary follows Zipf's power law.
               AI text deviates: over-represents mid-frequency words.

  Signal 5  -- Yule's K Characteristic (Vocabulary Richness)
               Yule 1944 vocabulary measure. Lower K in AI text (narrower lexicon).

  Signal 6  -- Hapax Legomena Rate
               Proportion of words used exactly once.
               Human text has more unique words; AI text repeats more.

  Signal 7  -- N-gram Entropy
               Shannon entropy of character/word bigrams.
               AI text has lower entropy (more predictable local patterns).

  Signal 8  -- Stylometric Fingerprint
               Function-word frequency profile, punctuation patterns,
               sentence-opener diversity, paragraph length variance.

  Signal 9  -- Semantic Coherence Scoring
               Cosine similarity between adjacent sentence TF-IDF vectors.
               AI text is hyper-coherent; human text has natural topic drift.
"""

from __future__ import annotations

import math
import re
import statistics
import collections
from dataclasses import dataclass, field
from typing import Optional

from academicguard.core.document import Document
from academicguard.core.report import Finding, ModuleResult


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class SignalResult:
    name: str
    raw_value: float        # the actual measured value
    ai_probability: float   # 0.0 (human) -> 1.0 (AI)
    weight: float
    interpretation: str


@dataclass
class AIDetectionResult:
    overall_probability: float
    label: str              # LIKELY_AI | UNCERTAIN | LIKELY_HUMAN
    confidence: str         # high | medium | low
    signals: list[SignalResult]
    per_paragraph: list[dict]
    gltr_stats: dict        # green/yellow/red/purple zone counts


# ------------------------------------------------------------------ #
# Signal weights (tuned on academic text corpus)
# ------------------------------------------------------------------ #

SIGNAL_WEIGHTS = {
    "GLTR Token Rank":        0.22,
    "Perplexity":             0.18,
    "Burstiness":             0.12,
    "Zipf Deviation":         0.10,
    "Yule K":                 0.10,
    "Hapax Rate":             0.08,
    "N-gram Entropy":         0.08,
    "Stylometric Profile":    0.07,
    "Semantic Coherence":     0.05,
}


# ------------------------------------------------------------------ #
# GLTR -- Giant Language Model Test Room (MIT-IBM Watson AI Lab)
# ------------------------------------------------------------------ #

class GLTRAnalyzer:
    """
    Reimplementation of GLTR (Gehrmann et al., 2019).
    Reference: https://gltr.io / https://arxiv.org/abs/1906.04043

    Uses GPT-2 to rank each token's position in the model's probability
    distribution at that context point.

    Zones:
      Green  (rank 1-10)   -- highly predictable, very likely AI
      Yellow (rank 11-100) -- predictable
      Red    (rank 101-1000) -- less predictable
      Purple (rank >1000)  -- surprising, likely human

    AI text: heavily concentrated in green.
    Human text: spread across all zones.
    """

    def __init__(self, model_name: str = "gpt2"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._available = False

    def _load(self) -> bool:
        if self._model is not None:
            return self._available
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()
            self._torch = torch
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def analyze(self, text: str) -> dict:
        """
        Returns zone counts and green_fraction (primary AI signal).
        Falls back to heuristic if GPT-2 unavailable.
        """
        if not self._load():
            return self._heuristic_gltr(text)

        import torch
        tokens = self._tokenizer.encode(text, truncation=True, max_length=512)
        if len(tokens) < 5:
            return {"green": 0, "yellow": 0, "red": 0, "purple": 0,
                    "green_fraction": 0.5, "available": False}

        token_tensor = torch.tensor([tokens])
        zones = {"green": 0, "yellow": 0, "red": 0, "purple": 0}

        with torch.no_grad():
            outputs = self._model(token_tensor)
            logits = outputs.logits[0]  # (seq_len, vocab)

        for i in range(len(tokens) - 1):
            next_token = tokens[i + 1]
            probs = torch.softmax(logits[i], dim=-1)
            # rank of actual next token (1-indexed, 1 = most probable)
            rank = (probs > probs[next_token]).sum().item() + 1
            if rank <= 10:
                zones["green"] += 1
            elif rank <= 100:
                zones["yellow"] += 1
            elif rank <= 1000:
                zones["red"] += 1
            else:
                zones["purple"] += 1

        total = sum(zones.values()) or 1
        green_fraction = zones["green"] / total
        return {**zones, "green_fraction": green_fraction, "available": True}

    def _heuristic_gltr(self, text: str) -> dict:
        """
        Fallback when GPT-2 unavailable.
        Estimate green fraction via character-level bigram predictability.
        """
        words = text.lower().split()
        if not words:
            return {"green": 0, "yellow": 0, "red": 0, "purple": 0,
                    "green_fraction": 0.5, "available": False}

        # Use word frequency as proxy for rank
        freq = collections.Counter(words)
        total_words = len(words)
        sorted_by_freq = sorted(freq.values(), reverse=True)
        rank_map = {w: i + 1 for i, (w, _) in enumerate(freq.most_common())}

        zones = {"green": 0, "yellow": 0, "red": 0, "purple": 0}
        for w in words:
            r = rank_map.get(w, len(rank_map))
            if r <= 10:
                zones["green"] += 1
            elif r <= 100:
                zones["yellow"] += 1
            elif r <= 1000:
                zones["red"] += 1
            else:
                zones["purple"] += 1

        total = sum(zones.values()) or 1
        return {**zones, "green_fraction": zones["green"] / total, "available": False}

    def ai_probability(self, gltr_stats: dict) -> float:
        """
        Green fraction -> AI probability.
        Based on GLTR paper: >70% green strongly indicates AI.
        """
        gf = gltr_stats.get("green_fraction", 0.5)
        # Sigmoid-like mapping: 0.7 green -> ~0.9 AI prob
        return min(1.0, max(0.0, (gf - 0.30) / 0.45))


# ------------------------------------------------------------------ #
# Perplexity (sliding window)
# ------------------------------------------------------------------ #

class PerplexityAnalyzer:
    """
    GPT-2 perplexity with a sliding window to get paragraph-level scores.
    Lower perplexity = text is more predictable = more likely AI-generated.
    """

    def __init__(self, model_name: str = "gpt2"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()
            self._torch = torch
            return True
        except Exception:
            return False

    def perplexity(self, text: str) -> float:
        if not self._load():
            return self._heuristic_perplexity(text)
        import torch
        enc = self._tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = self._model(enc, labels=enc).loss.item()
        return math.exp(loss)

    def _heuristic_perplexity(self, text: str) -> float:
        """Word bigram cross-entropy as perplexity proxy."""
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        if len(words) < 10:
            return 200.0
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        freq = collections.Counter(bigrams)
        total = len(bigrams)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
        # map entropy to approximate perplexity
        return 2 ** entropy

    def ai_probability(self, ppl: float) -> float:
        """
        GPT-2 perplexity ranges:
          Human academic text: 100-600
          AI-generated text:   20-100
        Map to [0,1] probability of AI.
        """
        return min(1.0, max(0.0, 1.0 - (ppl - 15.0) / 400.0))


# ------------------------------------------------------------------ #
# Burstiness
# ------------------------------------------------------------------ #

class BurstinessAnalyzer:
    """
    Measures coefficient of variation (CV) of sentence lengths.
    Human writing is 'bursty' (high CV); AI writing is uniform (low CV).
    Reference: Guo et al. (2023) -- burstiness as AI detection feature.
    """

    def analyze(self, text: str) -> float:
        """Returns burstiness coefficient [0, inf). Higher = more human."""
        sentences = re.findall(r"[^.!?]+[.!?]+", text)
        lengths = [len(s.split()) for s in sentences if len(s.split()) >= 2]
        if len(lengths) < 5:
            return 0.5
        mean_l = statistics.mean(lengths)
        std_l = statistics.pstdev(lengths)
        return std_l / mean_l if mean_l > 0 else 0.0

    def ai_probability(self, burstiness: float) -> float:
        """
        CV < 0.25 -> very uniform -> high AI probability
        CV > 0.65 -> bursty -> low AI probability
        """
        return min(1.0, max(0.0, 1.0 - burstiness / 0.60))


# ------------------------------------------------------------------ #
# Zipf's Law Deviation
# ------------------------------------------------------------------ #

class ZipfAnalyzer:
    """
    Zipf's law: in natural language, word frequency ~ 1/rank.
    Human text follows this closely. AI text deviates -- mid-frequency
    words are over-represented (model tends toward 'safe' word choices).

    Method: compute R^2 of log(freq) vs log(rank) regression.
    Low R^2 = Zipfian deviation = higher AI probability.
    """

    def analyze(self, text: str) -> float:
        """Returns Zipf R^2 [0, 1]. Higher = better Zipfian fit = more human."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        if len(words) < 50:
            return 0.5
        freq = collections.Counter(words)
        ranked = sorted(freq.values(), reverse=True)
        n = min(100, len(ranked))
        log_ranks = [math.log(i + 1) for i in range(n)]
        log_freqs = [math.log(f) for f in ranked[:n]]
        return self._r_squared(log_ranks, log_freqs)

    def _r_squared(self, x: list, y: list) -> float:
        n = len(x)
        if n < 3:
            return 0.5
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot == 0:
            return 1.0
        ss_res = sum(
            (y[i] - (mean_y + (sum((x[j]-mean_x)*(y[j]-mean_y) for j in range(n)) /
                                 sum((x[j]-mean_x)**2 for j in range(n)) if sum((x[j]-mean_x)**2 for j in range(n)) else 1)
                     * (x[i] - mean_x))) ** 2
            for i in range(n)
        )
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    def ai_probability(self, r2: float) -> float:
        """Low R^2 (poor Zipf fit) -> high AI probability."""
        return min(1.0, max(0.0, 1.0 - r2))


# ------------------------------------------------------------------ #
# Yule's K Characteristic
# ------------------------------------------------------------------ #

class YuleKAnalyzer:
    """
    Yule's K (1944): vocabulary richness measure.
    K = 10^4 * (sum_i(i^2 * V(i,N)) - N) / N^2
    where V(i,N) = number of words occurring exactly i times.

    Lower K -> richer vocabulary -> more human-like.
    Higher K -> less varied lexicon -> AI-like.
    Reference: Tweedie & Baayen (1998).
    """

    def analyze(self, text: str) -> float:
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        N = len(words)
        if N < 50:
            return 100.0
        freq = collections.Counter(words)
        freq_of_freq = collections.Counter(freq.values())
        M1 = N
        M2 = sum(i * i * v for i, v in freq_of_freq.items())
        if M1 == 0:
            return 100.0
        K = 10000 * (M2 - M1) / (M1 * M1)
        return max(0.0, K)

    def ai_probability(self, K: float) -> float:
        """
        Human academic text: K ~ 50-150
        AI text: K ~ 150-300 (less varied, repetitive)
        """
        return min(1.0, max(0.0, (K - 80.0) / 200.0))


# ------------------------------------------------------------------ #
# Hapax Legomena Rate
# ------------------------------------------------------------------ #

class HapaxAnalyzer:
    """
    Hapax legomena: words that appear exactly once.
    High hapax rate = rich vocabulary = more human.
    Low hapax rate = repetitive = AI-like.
    """

    def analyze(self, text: str) -> float:
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        if len(words) < 20:
            return 0.5
        freq = collections.Counter(words)
        hapax = sum(1 for f in freq.values() if f == 1)
        return hapax / len(freq)

    def ai_probability(self, hapax_rate: float) -> float:
        """High hapax rate -> human. Low -> AI."""
        return min(1.0, max(0.0, 1.0 - hapax_rate * 1.4))


# ------------------------------------------------------------------ #
# N-gram Entropy
# ------------------------------------------------------------------ #

class NGramEntropyAnalyzer:
    """
    Shannon entropy of word bigrams and character trigrams.
    AI text has lower entropy (more predictable local patterns).
    Human text has higher entropy (more lexical diversity).
    """

    def analyze(self, text: str) -> dict:
        word_entropy = self._word_bigram_entropy(text)
        char_entropy = self._char_trigram_entropy(text)
        return {"word_bigram": word_entropy, "char_trigram": char_entropy}

    def _word_bigram_entropy(self, text: str) -> float:
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        if len(words) < 10:
            return 5.0
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return self._entropy(bigrams)

    def _char_trigram_entropy(self, text: str) -> float:
        clean = re.sub(r"\s+", " ", text.lower())
        if len(clean) < 30:
            return 8.0
        trigrams = [clean[i:i+3] for i in range(len(clean)-2)]
        return self._entropy(trigrams)

    def _entropy(self, items: list) -> float:
        if not items:
            return 0.0
        total = len(items)
        freq = collections.Counter(items)
        return -sum((c/total) * math.log2(c/total) for c in freq.values())

    def ai_probability(self, stats: dict) -> float:
        """
        Word bigram entropy:
          Human: 8-12 bits  |  AI: 4-7 bits
        Char trigram entropy:
          Human: 6-9 bits   |  AI: 4-6 bits
        """
        wb = stats.get("word_bigram", 8.0)
        ct = stats.get("char_trigram", 7.0)
        wb_prob = min(1.0, max(0.0, 1.0 - (wb - 3.0) / 9.0))
        ct_prob = min(1.0, max(0.0, 1.0 - (ct - 3.0) / 6.0))
        return 0.6 * wb_prob + 0.4 * ct_prob


# ------------------------------------------------------------------ #
# Stylometric Fingerprint
# ------------------------------------------------------------------ #

class StylometricAnalyzer:
    """
    Stylometric features used in authorship attribution and AI detection.
    Based on Mosteller & Wallace (1964) and modern AI detection research.

    Features:
    - Function word frequency profile (AI overuses specific function words)
    - Sentence-opener diversity (AI: "The/This/These" dominate)
    - Punctuation density and variety
    - Paragraph length coefficient of variation
    - Known AI filler phrase density
    """

    # AI-characteristic function words (overused vs human baseline)
    _AI_FUNCTION_WORDS = [
        "furthermore", "moreover", "additionally", "consequently",
        "nevertheless", "nonetheless", "however", "therefore",
        "thus", "hence", "subsequently", "accordingly",
    ]

    # AI filler phrases (from GLTR research + manual curation)
    _AI_PHRASES = [
        r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b",
        r"\boverall\b.*\b(approach|method|results)\b",
        r"\bnovel (approach|method|framework|algorithm)\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
        r"\bsignificant(ly)? (improve|outperform|reduce|increase)\b",
        r"\brobust(ness)?\b", r"\bseamlessly?\b", r"\bcomprehensive(ly)?\b",
        r"\bleverag(e|ing|ed)\b", r"\bdelve\b", r"\bunlock\b",
        r"\bit is (worth|important) (noting|to note)\b",
        r"\bit should be noted\b", r"\bit is evident\b",
        r"\bthe proposed (method|approach|framework|model|system)\b",
        r"\bthe experimental results (show|demonstrate|indicate|suggest)\b",
        r"\bout?perform(s|ed)? (all|existing|state|baseline)\b",
    ]

    def analyze(self, text: str) -> dict:
        words = text.split()
        word_count = max(1, len(words))
        sentences = re.findall(r"[^.!?]+[.!?]+", text)

        # Function word overuse
        fw_density = sum(
            len(re.findall(rf"\b{fw}\b", text.lower()))
            for fw in self._AI_FUNCTION_WORDS
        ) / (word_count / 100)

        # AI filler phrase density
        phrase_hits = sum(
            len(re.findall(p, text.lower())) for p in self._AI_PHRASES
        )
        phrase_density = phrase_hits / (word_count / 100)

        # Sentence opener diversity
        openers = [s.strip().split()[0].lower() if s.strip().split() else ""
                   for s in sentences[:30]]
        opener_diversity = len(set(openers)) / max(1, len(openers))

        # Punctuation variety
        punct_chars = re.findall(r"[,;:\-\(\)\[\]\"']", text)
        punct_variety = len(set(punct_chars)) / max(1, len(punct_chars)) if punct_chars else 0

        # Paragraph length variance
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        para_lengths = [len(p.split()) for p in paragraphs]
        if len(para_lengths) >= 3:
            para_cv = statistics.pstdev(para_lengths) / max(1, statistics.mean(para_lengths))
        else:
            para_cv = 0.5

        return {
            "fw_density": fw_density,
            "phrase_density": phrase_density,
            "opener_diversity": opener_diversity,
            "punct_variety": punct_variety,
            "para_cv": para_cv,
        }

    def ai_probability(self, stats: dict) -> float:
        # Higher phrase density and function word density -> AI
        phrase_signal = min(1.0, stats["phrase_density"] / 4.0)
        fw_signal = min(1.0, stats["fw_density"] / 3.0)
        # Lower opener diversity -> AI (always starts same way)
        opener_signal = min(1.0, max(0.0, 1.0 - stats["opener_diversity"] * 1.5))
        # Lower punctuation variety -> AI
        punct_signal = min(1.0, max(0.0, 1.0 - stats["punct_variety"] * 2.0))
        # Lower paragraph CV -> AI
        para_signal = min(1.0, max(0.0, 1.0 - stats["para_cv"] * 2.0))

        return (
            0.30 * phrase_signal +
            0.25 * fw_signal +
            0.20 * opener_signal +
            0.15 * punct_signal +
            0.10 * para_signal
        )


# ------------------------------------------------------------------ #
# Semantic Coherence
# ------------------------------------------------------------------ #

class SemanticCoherenceAnalyzer:
    """
    Measures cosine similarity between adjacent sentences using TF-IDF vectors.
    AI text is hyper-coherent (every sentence directly follows the last).
    Human academic text has natural topic variation and occasional drift.

    Method: TF-IDF vectorization (no external models needed).
    """

    def analyze(self, text: str) -> float:
        """Returns mean cosine similarity between adjacent sentences [0, 1]."""
        sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+", text)
                     if len(s.split()) >= 5]
        if len(sentences) < 4:
            return 0.5

        # Build TF-IDF vocabulary from all sentences
        vocab = self._build_vocab(sentences)
        vectors = [self._tfidf_vector(s, vocab, sentences) for s in sentences]

        # Compute cosine similarity between adjacent sentences
        sims = [
            self._cosine(vectors[i], vectors[i+1])
            for i in range(len(vectors)-1)
        ]
        return statistics.mean(sims) if sims else 0.5

    def _build_vocab(self, sentences: list) -> dict:
        all_words = set()
        for s in sentences:
            all_words.update(re.findall(r"\b[a-z]{3,}\b", s.lower()))
        return {w: i for i, w in enumerate(sorted(all_words))}

    def _tfidf_vector(self, sentence: str, vocab: dict, all_sentences: list) -> list:
        words = re.findall(r"\b[a-z]{3,}\b", sentence.lower())
        n = len(vocab)
        tf = [0.0] * n
        total = max(1, len(words))
        for w in words:
            if w in vocab:
                tf[vocab[w]] += 1.0 / total
        # IDF
        N = len(all_sentences)
        vec = []
        for w, idx in vocab.items():
            df = sum(1 for s in all_sentences if w in s.lower())
            idf = math.log((N + 1) / (df + 1)) + 1
            vec.append(tf[idx] * idf)
        return vec

    def _cosine(self, v1: list, v2: list) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = math.sqrt(sum(a*a for a in v1))
        m2 = math.sqrt(sum(b*b for b in v2))
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)

    def ai_probability(self, coherence: float) -> float:
        """
        High coherence (>0.7) -> AI. Natural text: 0.2-0.5 mean similarity.
        """
        return min(1.0, max(0.0, (coherence - 0.20) / 0.55))


# ------------------------------------------------------------------ #
# Main AIDetector (ensemble)
# ------------------------------------------------------------------ #

class AIDetector:
    """
    Ensemble AI content detector -- fully self-contained, no external APIs.
    Combines 9 independent signals with empirically tuned weights.
    """

    def __init__(self, use_transformer: bool = True, model_name: str = "gpt2"):
        self._use_transformer = use_transformer
        self._model_name = model_name
        self._gltr = GLTRAnalyzer(model_name)
        self._perplexity = PerplexityAnalyzer(model_name)
        self._burstiness = BurstinessAnalyzer()
        self._zipf = ZipfAnalyzer()
        self._yule = YuleKAnalyzer()
        self._hapax = HapaxAnalyzer()
        self._ngram = NGramEntropyAnalyzer()
        self._stylo = StylometricAnalyzer()
        self._coherence = SemanticCoherenceAnalyzer()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, doc: Document) -> ModuleResult:
        text = (doc.abstract + "\n\n" + doc.body_text).strip() or doc.raw_text
        result = self._detect(text)

        score = 1.0 - result.overall_probability
        label = "PASS" if score >= 0.70 else ("WARN" if score >= 0.45 else "FAIL")

        return ModuleResult(
            module="AI Detector",
            score=score,
            label=label,
            summary=self._summary(result),
            findings=self._build_findings(result),
            metadata={
                "verdict": result.label,
                "overall_ai_probability": round(result.overall_probability, 3),
                "confidence": result.confidence,
                "gltr_green_fraction": round(result.gltr_stats.get("green_fraction", 0), 3),
                "signals": {s.name: round(s.ai_probability, 3) for s in result.signals},
            },
        )

    def analyze_text(self, text: str) -> AIDetectionResult:
        return self._detect(text)

    # ------------------------------------------------------------------ #
    # Detection pipeline
    # ------------------------------------------------------------------ #

    def _detect(self, text: str) -> AIDetectionResult:
        if len(text.split()) < 30:
            return AIDetectionResult(
                overall_probability=0.5, label="UNCERTAIN", confidence="low",
                signals=[], per_paragraph=[], gltr_stats={},
            )

        signals: list[SignalResult] = []

        # Signal 1: GLTR
        gltr_stats = self._gltr.analyze(text)
        gf = gltr_stats.get("green_fraction", 0.5)
        signals.append(SignalResult(
            name="GLTR Token Rank",
            raw_value=gf,
            ai_probability=self._gltr.ai_probability(gltr_stats),
            weight=SIGNAL_WEIGHTS["GLTR Token Rank"],
            interpretation=f"Green zone fraction: {gf:.1%} (>60% strongly indicates AI)",
        ))

        # Signal 2: Perplexity
        ppl = self._perplexity.perplexity(text)
        signals.append(SignalResult(
            name="Perplexity",
            raw_value=ppl,
            ai_probability=self._perplexity.ai_probability(ppl),
            weight=SIGNAL_WEIGHTS["Perplexity"],
            interpretation=f"GPT-2 perplexity: {ppl:.1f} (human ~200-500, AI ~20-100)",
        ))

        # Signal 3: Burstiness
        burst = self._burstiness.analyze(text)
        signals.append(SignalResult(
            name="Burstiness",
            raw_value=burst,
            ai_probability=self._burstiness.ai_probability(burst),
            weight=SIGNAL_WEIGHTS["Burstiness"],
            interpretation=f"Sentence length CV: {burst:.2f} (human >0.4, AI <0.25)",
        ))

        # Signal 4: Zipf
        zipf_r2 = self._zipf.analyze(text)
        signals.append(SignalResult(
            name="Zipf Deviation",
            raw_value=zipf_r2,
            ai_probability=self._zipf.ai_probability(zipf_r2),
            weight=SIGNAL_WEIGHTS["Zipf Deviation"],
            interpretation=f"Zipf R^2: {zipf_r2:.3f} (human ~0.9+, AI <0.75)",
        ))

        # Signal 5: Yule K
        K = self._yule.analyze(text)
        signals.append(SignalResult(
            name="Yule K",
            raw_value=K,
            ai_probability=self._yule.ai_probability(K),
            weight=SIGNAL_WEIGHTS["Yule K"],
            interpretation=f"Yule's K: {K:.1f} (human 50-150, AI 150-300)",
        ))

        # Signal 6: Hapax
        hapax = self._hapax.analyze(text)
        signals.append(SignalResult(
            name="Hapax Rate",
            raw_value=hapax,
            ai_probability=self._hapax.ai_probability(hapax),
            weight=SIGNAL_WEIGHTS["Hapax Rate"],
            interpretation=f"Hapax rate: {hapax:.1%} (human >40%, AI <25%)",
        ))

        # Signal 7: N-gram entropy
        ngram_stats = self._ngram.analyze(text)
        signals.append(SignalResult(
            name="N-gram Entropy",
            raw_value=ngram_stats["word_bigram"],
            ai_probability=self._ngram.ai_probability(ngram_stats),
            weight=SIGNAL_WEIGHTS["N-gram Entropy"],
            interpretation=f"Word bigram entropy: {ngram_stats['word_bigram']:.2f} bits (human >8, AI <6)",
        ))

        # Signal 8: Stylometric
        stylo_stats = self._stylo.analyze(text)
        signals.append(SignalResult(
            name="Stylometric Profile",
            raw_value=stylo_stats["phrase_density"],
            ai_probability=self._stylo.ai_probability(stylo_stats),
            weight=SIGNAL_WEIGHTS["Stylometric Profile"],
            interpretation=f"AI phrase density: {stylo_stats['phrase_density']:.2f}/100w, "
                           f"opener diversity: {stylo_stats['opener_diversity']:.1%}",
        ))

        # Signal 9: Semantic coherence
        coherence = self._coherence.analyze(text)
        signals.append(SignalResult(
            name="Semantic Coherence",
            raw_value=coherence,
            ai_probability=self._coherence.ai_probability(coherence),
            weight=SIGNAL_WEIGHTS["Semantic Coherence"],
            interpretation=f"Adjacent sentence similarity: {coherence:.2f} (AI >0.65, human 0.2-0.5)",
        ))

        # Weighted ensemble
        total_weight = sum(s.weight for s in signals)
        overall_prob = sum(s.ai_probability * s.weight for s in signals) / total_weight

        # Label and confidence
        if overall_prob >= 0.70:
            label = "LIKELY_AI"
            confidence = "high" if overall_prob >= 0.85 else "medium"
        elif overall_prob >= 0.45:
            label = "UNCERTAIN"
            confidence = "low"
        else:
            label = "LIKELY_HUMAN"
            confidence = "high" if overall_prob <= 0.25 else "medium"

        # Per-paragraph analysis (subset of signals for speed)
        per_para = self._analyze_paragraphs(text)

        return AIDetectionResult(
            overall_probability=overall_prob,
            label=label,
            confidence=confidence,
            signals=signals,
            per_paragraph=per_para,
            gltr_stats=gltr_stats,
        )

    def _analyze_paragraphs(self, text: str) -> list[dict]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.split()) >= 30]
        results = []
        for i, para in enumerate(paragraphs[:15]):
            burst = self._burstiness.analyze(para)
            hapax = self._hapax.analyze(para)
            stylo = self._stylo.analyze(para)
            ngram = self._ngram.analyze(para)
            prob = (
                0.35 * self._burstiness.ai_probability(burst) +
                0.25 * self._hapax.ai_probability(hapax) +
                0.25 * self._stylo.ai_probability(stylo) +
                0.15 * self._ngram.ai_probability(ngram)
            )
            results.append({
                "paragraph": i + 1,
                "snippet": para[:100] + ("..." if len(para) > 100 else ""),
                "ai_probability": round(prob, 3),
                "burstiness": round(burst, 3),
                "hapax_rate": round(hapax, 3),
            })
        return results

    # ------------------------------------------------------------------ #
    # Findings and summary
    # ------------------------------------------------------------------ #

    def _build_findings(self, result: AIDetectionResult) -> list[Finding]:
        findings: list[Finding] = []

        if result.label == "LIKELY_AI":
            findings.append(Finding(
                category="ai",
                severity="error",
                message=f"Text has high AI-generation probability ({result.overall_probability:.0%}). "
                        f"Confidence: {result.confidence}.",
                suggestion="Substantially rewrite in your own voice with specific technical details, "
                           "domain expertise, and natural stylistic variation.",
                rule_id="AI-001",
            ))
        elif result.label == "UNCERTAIN":
            findings.append(Finding(
                category="ai",
                severity="warning",
                message=f"Mixed AI/human signals detected ({result.overall_probability:.0%} AI probability). "
                        f"Some sections may be AI-assisted.",
                suggestion="Review flagged paragraphs. Ensure all claims are grounded in your own analysis.",
                rule_id="AI-002",
            ))

        # Report strongest signals
        top_signals = sorted(result.signals, key=lambda s: s.ai_probability, reverse=True)
        for sig in top_signals[:3]:
            if sig.ai_probability >= 0.60:
                findings.append(Finding(
                    category="ai",
                    severity="warning",
                    message=f"High AI signal -- {sig.name}: {sig.interpretation}",
                    suggestion="This pattern is associated with AI generation. Review this aspect of your writing.",
                    rule_id=f"AI-SIG-{sig.name[:4].upper().replace(' ', '')}",
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
                    suggestion="Rewrite with specific data, original observations, and varied sentence structure.",
                    rule_id="AI-003",
                ))

        return findings

    def _summary(self, result: AIDetectionResult) -> str:
        pct = f"{result.overall_probability:.0%}"
        gltr_pct = f"{result.gltr_stats.get('green_fraction', 0):.0%}"
        if result.label == "LIKELY_AI":
            return (f"AI probability {pct} (confidence: {result.confidence}). "
                    f"GLTR green zone: {gltr_pct}. Text shows strong AI-generation patterns.")
        if result.label == "UNCERTAIN":
            return (f"AI probability {pct} -- mixed signals. "
                    f"GLTR green zone: {gltr_pct}. Manual review recommended.")
        return (f"AI probability {pct} (confidence: {result.confidence}). "
                f"GLTR green zone: {gltr_pct}. Text appears predominantly human-written.")

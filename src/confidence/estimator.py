"""
Derived OCR Confidence Estimator.
Derives defensive, mathematically grounded confidence signals when the OCR backend
does not directly output raw token probabilities.
"""

from typing import Dict, Any, Optional
import re
import math


class ConfidenceEstimator:
    """
    Estimates a derived confidence score from observable text-level properties:
    - Optical noise ratio (ratio of non-word / garbled glyphs)
    - Valid English lexical coverage
    - Token length plausibility
    - Repetitive hallucination detection
    """

    # Common English high-frequency word set for lexical verification
    COMMON_VOCAB = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
        "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
        "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give",
        "day", "most", "us", "is", "are", "was", "were", "been", "has", "had", "did", "does", "student", "answer",
        "question", "exam", "write", "writing", "story", "paragraph", "letter", "chart", "graph", "electricity"
    }

    @classmethod
    def estimate_derived_confidence(cls, text: str) -> Optional[float]:
        """
        Derive an estimated confidence score [0.0, 1.0] for the extracted text.
        Returns None if text is empty.
        """
        if not text or not text.strip():
            return None

        tokens = text.split()
        if not tokens:
            return None

        total_tokens = len(tokens)

        # 1. Optical Noise Penalty: Count tokens with weird punctuation/symbols
        clean_words = 0
        vocab_matches = 0
        noise_tokens = 0
        char_count = 0
        non_ascii_count = 0

        for t in tokens:
            char_count += len(t)
            # Count non-alphanumeric noise
            if re.search(r"[\^~`\|\{\}\[\]\<\>\\]", t):
                noise_tokens += 1
            
            clean_t = re.sub(r"[^\w]", "", t).lower()
            if clean_t:
                clean_words += 1
                if clean_t in cls.COMMON_VOCAB or len(clean_t) <= 2:
                    vocab_matches += 1

        # Calculate base components
        clean_word_ratio = clean_words / max(1, total_tokens)
        noise_ratio = noise_tokens / max(1, total_tokens)
        vocab_ratio = vocab_matches / max(1, clean_words)

        # 2. Token Length Distribution Sanity
        avg_word_len = char_count / max(1, total_tokens)
        len_penalty = 0.0
        if avg_word_len > 15.0 or avg_word_len < 2.0:
            len_penalty = 0.3

        # 3. Repetition penalty (detect VLM degeneration loop)
        token_freq = {}
        for t in tokens:
            token_freq[t] = token_freq.get(t, 0) + 1
        max_rep = max(token_freq.values()) if token_freq else 1
        rep_ratio = max_rep / max(1, total_tokens)
        rep_penalty = 0.4 if (rep_ratio > 0.3 and total_tokens > 10) else 0.0

        # Formulate composite confidence
        raw_score = (0.5 * clean_word_ratio) + (0.5 * vocab_ratio) - (0.4 * noise_ratio) - len_penalty - rep_penalty
        
        # Clamp to [0.05, 0.98]
        derived = max(0.05, min(0.98, raw_score))
        return float(round(derived, 4))

#!/usr/bin/env python3
"""
AI Emotion Detection (text-based)
----------------------------------
A small, lexicon-based emotion detector for English text.

This is a rule-based demo, NOT a trained machine-learning model. It scores
text against a curated word lexicon (data/lexicon.json) covering six basic
emotions -- joy, sadness, anger, fear, surprise, disgust -- and applies
simple negation handling (e.g. "not happy") to reduce/flip a matched word's
contribution.

Usage:
    python detect.py "I am so happy today!"
    python detect.py "I am not happy about this."
"""

import json
import re
import sys
from pathlib import Path

# Words that negate/flip the sentiment of a following emotion word when they
# appear shortly before it.
NEGATION_WORDS = {
    "not", "no", "never", "without", "hardly", "barely", "cannot",
    "isnt", "wasnt", "arent", "werent", "dont", "didnt", "doesnt",
    "cant", "couldnt", "wouldnt", "shouldnt", "aint", "none", "nobody",
    "nothing",
}

# How many tokens back to look for a negation cue before an emotion word.
NEGATION_WINDOW = 3

EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "disgust")

_LEXICON_PATH = Path(__file__).resolve().parent / "data" / "lexicon.json"


def load_lexicon(path: Path = _LEXICON_PATH) -> dict:
    """Load the emotion -> word-list lexicon from disk and invert it into a
    word -> emotion lookup map for fast scoring."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    word_to_emotion = {}
    for emotion, words in raw.items():
        for word in words:
            word_to_emotion[word.lower()] = emotion
    return word_to_emotion


def tokenize(text: str):
    """Lowercase and split text into simple word tokens.

    Contractions like "isn't" are normalized to "isnt" so they match the
    NEGATION_WORDS set without needing an apostrophe-aware tokenizer.
    """
    text = text.lower()
    text = text.replace("'", "")
    return re.findall(r"[a-z]+", text)


def detect_emotions(text: str, word_to_emotion: dict | None = None) -> dict:
    """Score `text` against the emotion lexicon.

    Returns a dict with:
        - "scores": {emotion: float} raw scores for every emotion
        - "dominant": name of the highest-scoring emotion (or "neutral"
          if no lexicon words were matched at all, or if the best score
          is <= 0)
    """
    if word_to_emotion is None:
        word_to_emotion = load_lexicon()

    tokens = tokenize(text)
    scores = {emotion: 0.0 for emotion in EMOTIONS}
    matched_any = False

    for i, token in enumerate(tokens):
        emotion = word_to_emotion.get(token)
        if emotion is None:
            continue

        matched_any = True

        window_start = max(0, i - NEGATION_WINDOW)
        preceding = tokens[window_start:i]
        negated = any(w in NEGATION_WORDS for w in preceding)

        if negated:
            # Negation flips the contribution: "not happy" pulls the score
            # for joy down instead of up.
            scores[emotion] -= 1.0
        else:
            scores[emotion] += 1.0

    if not matched_any:
        dominant = "neutral"
    else:
        best_emotion = max(EMOTIONS, key=lambda e: scores[e])
        dominant = best_emotion if scores[best_emotion] > 0 else "neutral"

    return {"scores": scores, "dominant": dominant}


def format_result(result: dict) -> str:
    lines = ["Emotion scores:"]
    for emotion in EMOTIONS:
        lines.append(f"  {emotion:9s}: {result['scores'][emotion]:+.1f}")
    lines.append(f"Dominant emotion: {result['dominant']}")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print('Usage: python detect.py "some sentence"')
        return 1

    text = " ".join(argv[1:])
    result = detect_emotions(text)
    print(f'Input: "{text}"')
    print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

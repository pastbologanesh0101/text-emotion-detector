# AI Emotion Detection (Text-Based)

A small, self-contained, lexicon-based emotion detector for English text.

> **Note:** This is a simple rule-based / lexicon-matching demo, **not a
> trained machine-learning model**. There is no neural network, no
> embeddings, and no training data — it works by matching words in the
> input text against a small curated lexicon of emotion-associated words
> and applying a basic negation rule. It's meant as a lightweight,
> readable, fully offline example of how a rule-based emotion classifier
> can work, not a state-of-the-art emotion classifier.

## What it does

Given a piece of text, the detector:

1. Tokenizes the text into lowercase words.
2. Looks up each word in a bundled lexicon (`data/lexicon.json`) covering
   six basic emotions: **joy, sadness, anger, fear, surprise, disgust**.
3. Adds +1 to an emotion's score for each matching word.
4. Applies simple negation handling: if a negation word (`not`, `no`,
   `never`, `without`, `hardly`, `barely`, `cannot`, `don't`, `isn't`, ...)
   appears within the 3 tokens before a matched emotion word, that word's
   contribution is flipped to -1 instead of +1. So "happy" scores `+1` for
   joy, but "not happy" scores `-1` for joy.
5. Reports the score for every emotion and the **dominant emotion** (the
   highest-scoring emotion, or `neutral` if nothing matched or every score
   is zero or negative).

## Project structure

```
text-emotion-detector/
├── data/
│   └── lexicon.json       # curated emotion -> word list lexicon
├── detect.py               # core detection logic + CLI entry point
├── tests/
│   └── test_detect.py      # unit tests
├── .github/workflows/
│   └── tests.yml            # CI: runs the test suite on push/PR
├── LICENSE
└── README.md
```

## Requirements

Pure Python standard library — no third-party dependencies, no
`requirements.txt` needed. Works with Python 3.11+ (also fine on most
earlier Python 3 versions since it only uses `json`, `re`, `pathlib`,
and `sys`).

## How to run

```bash
python detect.py "I am so happy and excited today!"
```

Example output:

```
Input: "I am so happy and excited today!"
Emotion scores:
  joy      : +2.0
  sadness  : +0.0
  anger    : +0.0
  fear     : +0.0
  surprise : +0.0
  disgust  : +0.0
Dominant emotion: joy
```

Negation example:

```bash
python detect.py "I am not happy about this."
```

```
Input: "I am not happy about this."
Emotion scores:
  joy      : -1.0
  sadness  : +0.0
  anger    : +0.0
  fear     : +0.0
  surprise : +0.0
  disgust  : +0.0
Dominant emotion: neutral
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

or, if you have `pytest` installed:

```bash
pytest tests/
```

The test suite checks unambiguous example sentences for each of the six
emotions, negation handling (score reduction/flip), and neutral/empty
text handling.

## Limitations

- Small hand-curated lexicon — many real-world words and slang won't be
  recognized.
- No handling of sarcasm, idioms, emojis, multi-word phrases, or context
  beyond a simple word-window negation check.
- No language support beyond English.
- Not trained on any dataset and not benchmarked against standard emotion
  classification corpora — it's a teaching/demo tool, not a production
  classifier.

## License

MIT — see [LICENSE](LICENSE).

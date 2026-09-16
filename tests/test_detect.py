import os
import sys
import unittest

# Make the project root importable when tests are run directly
# (e.g. `python -m unittest` from the repo root, or via pytest).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect import detect_emotions, load_lexicon  # noqa: E402


class TestEmotionDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = load_lexicon()

    def _detect(self, text):
        return detect_emotions(text, self.lexicon)

    def test_joy_sentence(self):
        result = self._detect("I am so happy and joyful today, this is wonderful!")
        self.assertEqual(result["dominant"], "joy")
        self.assertGreater(result["scores"]["joy"], 0)

    def test_sadness_sentence(self):
        result = self._detect("I feel so sad and heartbroken, I keep crying.")
        self.assertEqual(result["dominant"], "sadness")
        self.assertGreater(result["scores"]["sadness"], 0)

    def test_anger_sentence(self):
        result = self._detect("I am furious and enraged, this makes me so angry!")
        self.assertEqual(result["dominant"], "anger")
        self.assertGreater(result["scores"]["anger"], 0)

    def test_fear_sentence(self):
        result = self._detect("I am terrified and scared, this is so frightening.")
        self.assertEqual(result["dominant"], "fear")
        self.assertGreater(result["scores"]["fear"], 0)

    def test_surprise_sentence(self):
        result = self._detect("Wow, I am so surprised and shocked, this is unexpected!")
        self.assertEqual(result["dominant"], "surprise")
        self.assertGreater(result["scores"]["surprise"], 0)

    def test_disgust_sentence(self):
        result = self._detect("This is disgusting and revolting, it makes me feel sick.")
        self.assertEqual(result["dominant"], "disgust")
        self.assertGreater(result["scores"]["disgust"], 0)

    def test_negation_reduces_joy(self):
        plain = self._detect("I am happy about this.")
        negated = self._detect("I am not happy about this.")
        self.assertGreater(plain["scores"]["joy"], negated["scores"]["joy"])
        self.assertLess(negated["scores"]["joy"], 0)
        self.assertNotEqual(negated["dominant"], "joy")

    def test_negation_reduces_fear(self):
        plain = self._detect("I am scared of spiders.")
        negated = self._detect("I am never scared of spiders.")
        self.assertGreater(plain["scores"]["fear"], negated["scores"]["fear"])
        self.assertLess(negated["scores"]["fear"], 0)

    def test_neutral_text(self):
        result = self._detect("The table is made of wood and metal.")
        self.assertEqual(result["dominant"], "neutral")

    def test_empty_text(self):
        result = self._detect("")
        self.assertEqual(result["dominant"], "neutral")
        for emotion in result["scores"]:
            self.assertEqual(result["scores"][emotion], 0.0)


if __name__ == "__main__":
    unittest.main()

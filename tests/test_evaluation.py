import unittest

from evaluation.run import metrics


class MetricTests(unittest.TestCase):
    def test_duplicate_hits_do_not_inflate_recall(self):
        self.assertEqual(metrics(["a", "b"], ["x", "a", "a"]), (0.5, 0.5))

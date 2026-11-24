import math
import os
import random
import tempfile
import unittest

import bench
import shipgate
from shipgate import Slice, binom_tail, evaluate, paired_upper, tabulate


def rows(policy, label, n, base_wrong=0, cand_wrong=0, both_wrong=0):
    """n cases; base_wrong / cand_wrong are the cases only that model misses."""
    out = []
    for i in range(n):
        b = i < base_wrong + both_wrong
        c = i < both_wrong or base_wrong + both_wrong <= i < base_wrong + both_wrong + cand_wrong
        out.append({"policy": policy, "label": label,
                    "baseline": label != b, "candidate": label != c})
    return out


class TestStats(unittest.TestCase):
    def test_binom_tail_exact(self):
        self.assertAlmostEqual(binom_tail(0, 10), 1.0)
        self.assertAlmostEqual(binom_tail(10, 10), 1 / 1024)
        self.assertAlmostEqual(binom_tail(8, 10), 56 / 1024)
        self.assertEqual(binom_tail(11, 10), 0.0)

    def test_binom_tail_matches_comb(self):
        for n in (1, 7, 40, 300):
            for k in (0, n // 3, n // 2, n):
                want = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
                self.assertAlmostEqual(binom_tail(k, n), want, places=10)

    def test_upper_bound_shrinks_with_n(self):
        small, _ = paired_upper(Slice("p", "miss", 50, 5, 5, 2, 2))
        big, _ = paired_upper(Slice("p", "miss", 5000, 500, 500, 200, 200))
        self.assertGreater(small, big)
        self.assertGreater(big, 0)

    def test_upper_bound_defined_with_zero_discordant(self):
        up, se = paired_upper(Slice("p", "miss", 20, 0, 0, 0, 0))
        self.assertGreater(up, 0)
        self.assertGreater(se, 0)


class TestTabulate(unittest.TestCase):
    def test_counts(self):
        r = rows("a", False, 100, base_wrong=3, cand_wrong=5, both_wrong=2)
        (s,) = tabulate(r)
        self.assertEqual((s.kind, s.n, s.base_err, s.cand_err, s.base_only, s.cand_only),
                         ("false_positive", 100, 5, 7, 3, 5))

    def test_split_by_class_and_policy(self):
        r = rows("a", False, 10) + rows("a", True, 10) + rows("b", True, 10)
        self.assertEqual([(s.policy, s.kind) for s in tabulate(r)],
                         [("a", "false_positive"), ("a", "miss"), ("b", "miss")])


class TestGate(unittest.TestCase):
    def test_ship_when_clearly_better(self):
        r = rows("a", False, 2000, base_wrong=40) + rows("a", True, 800, base_wrong=60, both_wrong=20)
        self.assertEqual(evaluate(r).decision, "SHIP")

    def test_hold_on_false_positive_regression(self):
        r = rows("a", False, 2000, cand_wrong=40, both_wrong=20) + rows("a", True, 800, base_wrong=60)
        v = evaluate(r)
        self.assertEqual(v.decision, "HOLD")
        self.assertEqual([x.slice.kind for x in v.reasons], ["false_positive"])

    def test_regression_hidden_by_other_policies(self):
        good = []
        for p in "bcdef":
            good += rows(p, False, 1000, base_wrong=20) + rows(p, True, 300, base_wrong=30)
        bad = rows("a", False, 1000, cand_wrong=50, both_wrong=10) + rows("a", True, 300, base_wrong=30)
        allrows = good + bad
        self.assertTrue(bench.naive_overall(allrows))  # accuracy went up
        v = evaluate(allrows)
        self.assertEqual(v.decision, "HOLD")
        self.assertEqual(v.reasons[0].slice.policy, "a")

    def test_more_data_when_slice_is_small(self):
        r = rows("a", False, 2000, base_wrong=10) + rows("a", True, 15)
        v = evaluate(r)
        self.assertEqual(v.decision, "MORE_DATA")
        self.assertEqual(v.reasons[0].slice.kind, "miss")
        self.assertGreater(v.reasons[0].more_cases, 0)

    def test_more_cases_estimate_is_enough(self):
        r = rows("a", True, 40, base_wrong=2, cand_wrong=2)
        v = evaluate(r)
        self.assertEqual(v.decision, "MORE_DATA")
        k = int(math.ceil((40 + v.reasons[0].more_cases) / 40.0)) + 1
        self.assertEqual(evaluate(r * k).decision, "SHIP")

    def test_identical_models_ship_with_enough_data(self):
        r = rows("a", False, 3000, both_wrong=90) + rows("a", True, 1000, both_wrong=150)
        self.assertEqual(evaluate(r).decision, "SHIP")

    def test_tighter_margin_blocks_ship(self):
        r = rows("a", False, 500, base_wrong=3, cand_wrong=3)
        self.assertEqual(evaluate(r).decision, "SHIP")
        self.assertEqual(evaluate(r, margins={"false_positive": 0.005}).decision, "MORE_DATA")

    def test_bonferroni_limits_false_holds(self):
        rng = random.Random(1)
        holds = sum(evaluate(bench.simulate(bench.SCENARIOS["no_change"][0], 2400, rng)).decision == "HOLD"
                    for _ in range(100))
        self.assertLessEqual(holds, 5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            evaluate([])


class TestIO(unittest.TestCase):
    def test_load_and_report(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data = shipgate.load(os.path.join(here, "sample_decisions.jsonl"))
        v = evaluate(data)
        text = shipgate.report(v)
        self.assertIn("decision: " + v.decision, text)
        self.assertEqual(v.decision, "HOLD")

    def test_missing_field(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"policy": "a", "label": true}\n')
        try:
            with self.assertRaises(ValueError):
                shipgate.load(f.name)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import shipgate
import skformat
import synth

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = skformat.load_config(os.path.join(HERE, "gate_config.json"))


def ev(oid, labels=(), at="2026-11-01T10:00:00.000Z", typ="workflow.succeeded"):
    if typ == "workflow.failed":
        return {"type": typ, "namespace": "products", "id": oid,
                "request_id": "req_x", "error": "images[2]: fetch failed"}
    return {"type": typ, "namespace": "products", "id": oid,
            "request_id": "req_x", "processed_at": at,
            "output": {"actions": [], "labels": [{"label": l} for l in labels]}}


def dec(oid, policy, decision, agent="bpo_review", reviewer="reviewer_01"):
    return {"reference_id": oid, "content_type": "product_listing",
            "policy_id": policy, "decision": decision, "agent_type": agent,
            "reviewer_id": reviewer}


class TestDocumentedShapes(unittest.TestCase):
    def test_docs_example_event(self):
        # the example payload on docs.safetykit.com "Receiving Results"
        doc = json.loads("""{
          "type": "workflow.succeeded", "namespace": "products",
          "id": "product_12345",
          "request_id": "req_01h2m7qdmdjckc30e1mnq6xqfd",
          "processed_at": "2026-11-22T18:04:31.512Z",
          "output": {"actions": [],
                     "labels": [{"label": "policy.example_violation"}],
                     "fields": {"confidence": 0.92},
                     "label_changes": {"added": ["policy.example_violation"],
                                       "removed": []}}}""")
        cfg = dict(CFG, label_to_policy={"policy.example_violation": "example"})
        idx, st = skformat.index_events([doc], cfg)
        self.assertEqual(idx, {"product_12345": {"example"}})
        self.assertEqual(st["unmapped_labels"], {})

    def test_docs_example_decision(self):
        # field names from the Agent Decisions request body
        d = {"reference_id": "listing_12345", "content_type": "listing",
             "policy_id": "counterfeit", "decision": "violation",
             "agent_type": "appeal", "reviewer_id": "reviewer_123",
             "comment": "Reviewed against the configured policy."}
        truth, _ = skformat.resolve_truth([d], CFG)
        self.assertEqual(truth, {("listing_12345", "counterfeit"): True})


class TestEvents(unittest.TestCase):
    def test_latest_delivery_wins_regardless_of_order(self):
        new = ev("a", ["policy.counterfeit"], at="2026-11-01T12:00:00.000Z")
        old = ev("a", [], at="2026-11-01T11:00:00.000Z")
        for order in ([old, new], [new, old]):
            idx, st = skformat.index_events(order, CFG)
            self.assertEqual(idx["a"], {"counterfeit"})
            self.assertEqual(st["duplicates"], 1)

    def test_timezone_offsets_are_compared_as_instants(self):
        late = ev("a", ["policy.ip_violation"], at="2026-11-01T12:30:00+00:00")
        early = ev("a", [], at="2026-11-01T14:00:00+02:00")   # 12:00 UTC
        idx, _ = skformat.index_events([late, early], CFG)
        self.assertEqual(idx["a"], {"ip"})

    def test_exact_redelivery_is_harmless(self):
        e = ev("a", ["policy.dropship"])
        idx, st = skformat.index_events([e, dict(e), dict(e)], CFG)
        self.assertEqual(idx, {"a": {"dropship"}})
        self.assertEqual(st["duplicates"], 2)

    def test_failed_object_has_no_decision(self):
        idx, st = skformat.index_events(
            [ev("a", typ="workflow.failed"), ev("b")], CFG)
        self.assertIsNone(idx["a"])
        self.assertEqual(idx["b"], set())
        self.assertEqual(st["failed"], 1)

    def test_success_beats_failure_either_order(self):
        ok, bad = ev("a", ["policy.hazmat"]), ev("a", typ="workflow.failed")
        for order in ([ok, bad], [bad, ok]):
            idx, st = skformat.index_events(order, CFG)
            self.assertEqual(idx["a"], {"hazmat"})
            self.assertEqual(st["failed"], 0)

    def test_unmapped_labels_and_other_events_are_counted_not_used(self):
        events = [ev("a", ["quality.low_resolution_image", "policy.adult_content"]),
                  {"type": "live_video.violation_detected", "id": "v1"}]
        idx, st = skformat.index_events(events, CFG)
        self.assertEqual(idx, {"a": {"adult"}})
        self.assertEqual(st["unmapped_labels"], {"quality.low_resolution_image": 1})
        self.assertEqual(st["other_types"], 1)

    def test_malformed_events_raise(self):
        no_id = ev("a")
        del no_id["id"]
        no_out = ev("a")
        del no_out["output"]
        bad_label = ev("a")
        bad_label["output"]["labels"] = [{"explanation": "no id"}]
        for e in (no_id, no_out, bad_label):
            with self.assertRaises(ValueError):
                skformat.index_events([e], CFG)

    def test_unparseable_time_loses_to_a_dated_delivery(self):
        dated = ev("a", ["policy.counterfeit"])
        junk = ev("a", [], at="yesterday")
        for order in ([dated, junk], [junk, dated]):
            idx, _ = skformat.index_events(order, CFG)
            self.assertEqual(idx["a"], {"counterfeit"}, order)


class TestTruth(unittest.TestCase):
    def test_higher_ranked_review_overrides(self):
        ds = [dec("a", "ip", "violation"),
              dec("a", "ip", "no_violation", agent="policy_expert")]
        for order in (ds, ds[::-1]):
            truth, st = skformat.resolve_truth(order, CFG)
            self.assertEqual(truth, {("a", "ip"): False})
            self.assertEqual(st["overridden"], 1)

    def test_same_rank_disagreement_is_contested(self):
        ds = [dec("a", "ip", "violation", reviewer="r1"),
              dec("a", "ip", "no_violation", reviewer="r2"),
              dec("b", "ip", "violation", reviewer="r1"),
              dec("b", "ip", "violation", reviewer="r2")]
        truth, st = skformat.resolve_truth(ds, CFG)
        self.assertEqual(truth, {("b", "ip"): True})
        self.assertEqual(st["contested"], 1)
        self.assertEqual(st["contested_by_policy"], {"ip": 1})

    def test_expert_settles_a_vendor_disagreement(self):
        ds = [dec("a", "ip", "violation", reviewer="r1"),
              dec("a", "ip", "no_violation", reviewer="r2"),
              dec("a", "ip", "violation", agent="policy_expert")]
        truth, st = skformat.resolve_truth(ds, CFG)
        self.assertEqual(truth, {("a", "ip"): True})
        self.assertEqual(st["contested"], 0)

    def test_same_object_two_policies_are_separate_cases(self):
        ds = [dec("a", "ip", "violation"), dec("a", "counterfeit", "no_violation")]
        truth, _ = skformat.resolve_truth(ds, CFG)
        self.assertEqual(truth, {("a", "ip"): True, ("a", "counterfeit"): False})

    def test_unknown_verdict_raises(self):
        with self.assertRaises(ValueError) as cm:
            skformat.resolve_truth([dec("a", "ip", "escalate")], CFG)
        self.assertIn("escalate", str(cm.exception))

    def test_missing_required_field_raises(self):
        d = dec("a", "ip", "violation")
        del d["policy_id"]
        with self.assertRaises(ValueError):
            skformat.resolve_truth([d], CFG)

    def test_policy_outside_config_is_counted_and_skipped(self):
        truth, st = skformat.resolve_truth(
            [dec("a", "spam", "violation"), dec("a", "ip", "violation")], CFG)
        self.assertEqual(list(truth), [("a", "ip")])
        self.assertEqual(st["unknown_policy"], 1)

    def test_missing_agent_type_ranks_last(self):
        d = dec("a", "ip", "violation")
        del d["agent_type"]
        truth, _ = skformat.resolve_truth(
            [d, dec("a", "ip", "no_violation", agent="bpo_review")], CFG)
        self.assertEqual(truth, {("a", "ip"): False})


class TestJoin(unittest.TestCase):
    def test_rows(self):
        base = [ev("a", ["policy.ip_violation"]), ev("b"), ev("c")]
        cand = [ev("a"), ev("b", ["policy.ip_violation", "policy.counterfeit"]),
                ev("c", typ="workflow.failed")]
        ds = [dec("a", "ip", "violation"), dec("b", "ip", "no_violation"),
              dec("c", "ip", "violation"), dec("zzz", "ip", "violation")]
        rows, st = skformat.join(base, cand, ds, CFG)
        got = {r["id"]: (r["label"], r["baseline"], r["candidate"]) for r in rows}
        self.assertEqual(got, {"a": (True, True, False), "b": (False, False, True)})
        self.assertEqual((st["candidate_missing"], st["baseline_missing"],
                          st["both_missing"]), (1, 0, 1))
        self.assertAlmostEqual(st["candidate_coverage"], 2 / 3.0)

    def test_clean_synthetic_data_round_trips(self):
        rng = random.Random(3)
        cand = synth.scenarios(synth.DEV)["fp_regression_large"][0]
        b, c, d, clean = synth.generate(synth.DEV, cand, 3000, rng)
        rows, st = skformat.join(b, c, d, CFG)
        key = lambda r: r["id"]
        self.assertEqual(sorted(rows, key=key), sorted(clean, key=key))
        self.assertEqual(st["candidate_coverage"], 1.0)

    def test_messy_synthetic_data_matches_clean_where_labels_agree(self):
        rng = random.Random(4)
        cand = synth.scenarios(synth.HELDOUT)["no_change"][0]
        b, c, d, clean = synth.generate(synth.HELDOUT, cand, 6000, rng)
        self.assertGreater(len(b), 6000)          # duplicates were injected
        rows, st = skformat.join(b, c, d, CFG)
        self.assertGreater(st["truth"]["contested"], 0)
        self.assertGreater(st["truth"]["overridden"], 0)
        self.assertGreater(st["candidate"]["failed"], 0)
        self.assertLess(len(rows), 6000)
        by_id = {r["id"]: r for r in clean}
        wrong_label = 0
        for r in rows:
            t = by_id[r["id"]]
            # model decisions must survive duplicates and reordering exactly
            self.assertEqual((r["baseline"], r["candidate"], r["policy"]),
                             (t["baseline"], t["candidate"], t["policy"]))
            wrong_label += r["label"] != t["label"]
        # review noise is 4%; expert re-review and contested removal must
        # leave clearly less than that in the joined rows
        self.assertLess(wrong_label / float(len(rows)), 0.035)


class TestGate(unittest.TestCase):
    def data(self, scenario, n=19200, seed=11, settings=synth.DEV):
        rng = random.Random(seed)
        cand = synth.scenarios(settings)[scenario][0]
        return synth.generate(settings, cand, n, rng)[:3]

    def test_ship_and_hold_through_the_full_path(self):
        self.assertEqual(skformat.gate(*self.data("better_everywhere"), cfg=CFG).decision, "SHIP")
        rv = skformat.gate(*self.data("fp_regression_large"), cfg=CFG)
        self.assertEqual(rv.decision, "HOLD")
        self.assertEqual([(r.slice.policy, r.slice.kind) for r in rv.verdict.reasons],
                         [("adult", "false_positive")])

    def test_candidate_that_fails_on_hard_cases_is_held(self):
        b, c, d = self.data("better_everywhere")
        # the candidate errors out on 5% of objects instead of answering
        failed = set(e["id"] for e in c[::20])
        c = [ev(e["id"], typ="workflow.failed") if e["id"] in failed else e for e in c]
        rv = skformat.gate(b, c, d, CFG)
        self.assertEqual(rv.verdict.decision, "SHIP")   # slices alone look fine
        self.assertEqual(rv.decision, "HOLD")
        self.assertIn("coverage", rv.notes[0])
        self.assertIn("decision: HOLD (coverage", skformat.report_run(rv))

    def test_baseline_failures_do_not_count_against_candidate(self):
        b, c, d = self.data("better_everywhere")
        failed = set(e["id"] for e in b[::20])
        b = [ev(e["id"], typ="workflow.failed") if e["id"] in failed else e for e in b]
        rv = skformat.gate(b, c, d, CFG)
        self.assertEqual(rv.stats["candidate_coverage"], 1.0)
        self.assertNotEqual(rv.decision, "HOLD")

    def test_no_overlap_raises_with_a_hint(self):
        with self.assertRaises(ValueError) as cm:
            skformat.gate([ev("a")], [ev("a")], [dec("other", "ip", "violation")], CFG)
        self.assertIn("reference_id", str(cm.exception))

    def test_config_margins_are_used(self):
        b, c, d = self.data("better_everywhere", n=4800)
        loose = dict(CFG, margins={"false_positive": 0.2, "miss": 0.5})
        self.assertEqual(skformat.gate(b, c, d, CFG).decision, "MORE_DATA")
        self.assertEqual(skformat.gate(b, c, d, loose).decision, "SHIP")


class TestConfigAndCLI(unittest.TestCase):
    def write(self, d, name, obj_or_lines):
        p = os.path.join(d, name)
        with open(p, "w") as f:
            if isinstance(obj_or_lines, list):
                f.write("\n".join(json.dumps(o) for o in obj_or_lines) + "\n")
            else:
                f.write(obj_or_lines)
        return p

    def test_config_rejects_unknown_keys_and_overlapping_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                skformat.load_config(self.write(d, "c.json", '{"margin": 1}'))
            with self.assertRaises(ValueError):
                skformat.load_config(self.write(
                    d, "c.json", '{"violating_decisions": ["x"], "benign_decisions": ["x"]}'))

    def test_bad_json_line_names_the_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "e.jsonl", '{"type": "workflow.succeeded"}\n{oops\n')
            with self.assertRaises(ValueError) as cm:
                skformat.read_jsonl(p)
            self.assertIn("line 2", str(cm.exception))

    def test_cli_exit_codes(self):
        rng = random.Random(5)
        gate = os.path.join(HERE, "shipgate.py")
        cfg = os.path.join(HERE, "gate_config.json")
        with tempfile.TemporaryDirectory() as d:
            for scenario, code in (("better_everywhere", 0), ("miss_regression_large", 1)):
                cand = synth.scenarios(synth.DEV)[scenario][0]
                b, c, t, _ = synth.generate(synth.DEV, cand, 19200, rng)
                args = [sys.executable, gate,
                        "--baseline", self.write(d, "b.jsonl", b),
                        "--candidate", self.write(d, "c.jsonl", c),
                        "--truth", self.write(d, "t.jsonl", t), "--config", cfg]
                r = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(r.returncode, code, r.stdout)
                self.assertIn("candidate coverage 100.0%", r.stdout)
            bad = self.write(d, "bad.jsonl", [dec("a", "ip", "maybe")])
            r = subprocess.run(args[:7] + [bad], stdout=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("maybe", r.stdout)
            r = subprocess.run([sys.executable, gate, "--baseline", "x"],
                               stdout=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(r.returncode, 2)


class TestStatsRegression(unittest.TestCase):
    def test_fast_binom_tail_matches_exact_sum_everywhere(self):
        import math
        for n in (1, 2, 9, 10, 57, 400, 2001):
            for k in range(0, n + 2, max(1, n // 23)):
                want = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
                self.assertAlmostEqual(shipgate.binom_tail(k, n), want, places=10)

    def test_tiny_tails_keep_relative_accuracy(self):
        import math
        want = sum(math.comb(400, i) for i in range(300, 401)) / 2 ** 400
        got = shipgate.binom_tail(300, 400)
        self.assertLess(abs(got - want) / want, 1e-9)


class TestScoreBound(unittest.TestCase):
    def test_reduces_to_mcnemar_at_zero(self):
        self.assertAlmostEqual(shipgate.score_stat(18, 10, 160, 0.0),
                               (18 - 10) / 28 ** 0.5)

    def test_null_variance_uses_the_constrained_mle(self):
        # brute force: maximise the trinomial likelihood over the baseline-only
        # rate q with candidate-only rate fixed at q + delta, then compare the
        # statistic built from that q with score_stat
        import math
        for b, c, n, delta in ((18, 10, 160, 0.05), (3, 9, 400, 0.02),
                               (0, 0, 50, 0.05), (7, 0, 90, 0.10)):
            best, best_ll = None, None
            steps = 200000
            for i in range(1, steps):
                q = i * (1 - delta) / 2.0 / steps
                rest = 1 - 2 * q - delta
                if rest <= 0:
                    break
                ll = (b * math.log(q + delta) + c * math.log(q)
                      + (n - b - c) * math.log(rest))
                if best_ll is None or ll > best_ll:
                    best, best_ll = q, ll
            want = (b - c - n * delta) / math.sqrt(
                n * (2 * best + delta * (1 - delta)))
            self.assertAlmostEqual(shipgate.score_stat(b, c, n, delta), want,
                                   places=3)

    def test_upper_bound_solves_the_score_equation(self):
        s = shipgate.Slice("p", "miss", 160, 0, 0, 10, 18)
        up, se = shipgate.paired_upper(s)
        self.assertAlmostEqual(shipgate.score_stat(18, 10, 160, up),
                               -shipgate.Z95, places=6)
        self.assertGreater(up, (18 - 10) / 160.0)
        self.assertGreater(se, 0)

    def test_bound_stays_in_range_at_the_extremes(self):
        for s in (shipgate.Slice("p", "miss", 5, 0, 5, 0, 5),
                  shipgate.Slice("p", "miss", 5, 5, 0, 5, 0),
                  shipgate.Slice("p", "miss", 1, 0, 0, 0, 0)):
            up, _ = shipgate.paired_upper(s)
            self.assertLessEqual(up, 1.0)
            self.assertGreaterEqual(up, (s.cand_only - s.base_only) / float(s.n))

    def test_false_ship_rate_at_margin_is_near_five_percent(self):
        import bench
        saved = bench.CAL_CASES
        bench.CAL_CASES = [(160, 0.03, 0.05, 0.02), (160, 0.15, 0.20, 0.05)]
        try:
            res = bench.calibrate(trials=2000, seed=99)
        finally:
            bench.CAL_CASES = saved
        for r in res:
            # 2000 trials: 5% has a standard error of about 0.5 points
            self.assertLess(r["score_false_ship"], 0.065)
            self.assertLessEqual(r["score_false_ship"], r["wald_false_ship"])


if __name__ == "__main__":
    unittest.main()

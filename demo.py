"""demo: one HOLD the old overall-accuracy rule would have shipped, on messy
SYNTHETIC data, end to end in a few seconds.

  python demo.py

Writes three files to a temp directory in the shapes skformat reads (two
runs of webhook-style result events and one file of reviewer decisions),
then runs the same code path as the command line.
"""
import json
import os
import random
import tempfile

import bench
import skformat
import synth

HERE = os.path.dirname(os.path.abspath(__file__))


def write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    rng = random.Random(2027)
    settings = synth.HELDOUT
    cand = synth.scenarios(settings)["fp_regression_large"][0]
    base_ev, cand_ev, decisions, _ = synth.generate(settings, cand, 40000, rng)

    with tempfile.TemporaryDirectory() as d:
        paths = [os.path.join(d, n) for n in (
            "baseline_events.jsonl", "candidate_events.jsonl",
            "agent_decisions.jsonl")]
        for p, recs in zip(paths, (base_ev, cand_ev, decisions)):
            write(p, recs)
        print("SYNTHETIC data. The candidate is better on five policies and "
              "has 6 points more\nfalse positives on the smallest one "
              "(recalled).\n")
        print("one result event:   " + json.dumps(
            next(e for e in cand_ev if any(
                l["label"].startswith("policy.")
                for l in e.get("output", {}).get("labels", [])))))
        print("one ground truth:   " + json.dumps(decisions[0]) + "\n")
        cfg_path = os.path.join(HERE, "gate_config.json")
        rv = skformat.gate_files(paths[0], paths[1], paths[2], cfg_path)
        rows, _ = skformat.join(base_ev, cand_ev, decisions,
                                skformat.load_config(cfg_path))

    b = sum(r["baseline"] == r["label"] for r in rows) / float(len(rows))
    c = sum(r["candidate"] == r["label"] for r in rows) / float(len(rows))
    print("overall accuracy: baseline %.2f%%, candidate %.2f%% -> an overall "
          "accuracy rule says %s\n" % (100 * b, 100 * c,
                                       "SHIP" if bench.naive_overall(rows)
                                       else "HOLD"))
    print(skformat.report_run(rv))


if __name__ == "__main__":
    main()

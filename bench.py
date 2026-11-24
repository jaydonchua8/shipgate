"""Simulate candidate models with known true quality and count how often each
ship rule makes the right call, on the dev setting and on the held-out one.

Every trial goes through the full path: synthetic webhook-shaped events and
reviewer decisions -> skformat.join -> shipgate.evaluate. All data is
synthetic. Also measures the confidence bound on its own
(calibrate). Prints three tables and writes bench_results.json.

  python bench.py [trials]     default 100 trials per row
"""
import json
import random
import sys

import skformat
import synth
from shipgate import Slice, evaluate, paired_upper, wald_upper

CFG = dict(skformat.DEFAULT_CONFIG,
           label_to_policy={v: k for k, v in synth.LABEL.items()},
           agent_type_priority=["policy_expert", "appeal", "bpo_review"])


def simulate(cand, n, rng, settings=None):
    """Rows for one simulated eval set, after the join."""
    b, c, d, _ = synth.generate(settings or synth.DEV, cand, n, rng)
    rows, _ = skformat.join(b, c, d, CFG)
    return rows


def naive_overall(rows):
    """Ship if overall accuracy did not go down."""
    b = sum(r["baseline"] == r["label"] for r in rows)
    c = sum(r["candidate"] == r["label"] for r in rows)
    return c >= b


def naive_slice(rows):
    """Ship only if no per-policy error count got worse."""
    worse = {}
    for r in rows:
        k = (r["policy"], r["label"])
        worse[k] = worse.get(k, 0) + (r["candidate"] != r["label"]) \
            - (r["baseline"] != r["label"])
    return all(v <= 0 for v in worse.values())


SETTINGS = {"dev": synth.DEV, "heldout": synth.HELDOUT}
SCENARIOS = synth.scenarios(synth.DEV)


def run_row(job):
    """One table row. Each row has its own seed so rows can run in parallel
    and still be reproduced one at a time."""
    setting, n, name, trials, seed = job
    settings = SETTINGS[setting]
    cand, should_ship = synth.scenarios(settings)[name]
    rng = random.Random("%s/%d/%s/%d" % (setting, n, name, seed))
    c = {"overall": 0, "slice": 0, "SHIP": 0, "HOLD": 0, "MORE_DATA": 0}
    for _ in range(trials):
        rows = simulate(cand, n, rng, settings)
        c["overall"] += naive_overall(rows)
        c["slice"] += naive_slice(rows)
        c[evaluate(rows).decision] += 1
    return {"setting": setting, "seed": seed, "cases": n,
            "scenario": name, "should_ship": should_ship, "trials": trials,
            "naive_overall_ship": c["overall"] / trials,
            "naive_slice_ship": c["slice"] / trials,
            "gate_ship": c["SHIP"] / trials,
            "gate_hold": c["HOLD"] / trials,
            "gate_more_data": c["MORE_DATA"] / trials}


def run(setting, trials, sizes, seed, workers=None):
    jobs = [(setting, n, name, trials, seed)
            for n in sizes for name in synth.scenarios(SETTINGS[setting])]
    if workers == 1:
        return [run_row(j) for j in jobs]
    import multiprocessing
    with multiprocessing.Pool(workers) as pool:
        return pool.map(run_row, jobs)


CAL_CASES = [  # slice size, baseline error, candidate error, margin
    (40, 0.15, 0.20, 0.05), (160, 0.15, 0.20, 0.05), (640, 0.15, 0.20, 0.05),
    (264, 0.25, 0.30, 0.05), (160, 0.03, 0.05, 0.02), (640, 0.03, 0.05, 0.02),
    (2560, 0.03, 0.05, 0.02)]


def calibrate(trials=4000, seed=5, rho=0.6):
    """One slice whose true regression is exactly the margin. A valid 95%
    bound clears the margin at most 5% of the time. Compares the version 1
    bound (adjusted Wald) with the score bound the gate uses now."""
    rng = random.Random(seed)
    out = []
    for n, eb, ec, margin in CAL_CASES:
        wald = score = 0
        for _ in range(trials):
            bo = co = 0
            for _ in range(n):
                u = rng.random()
                bw = u < eb
                cw = (u < ec) if rng.random() < rho else (rng.random() < ec)
                bo += bw and not cw
                co += cw and not bw
            s = Slice("p", "k", n, 0, 0, bo, co)
            wald += wald_upper(s)[0] < margin
            score += paired_upper(s)[0] < margin
        out.append({"slice_cases": n, "baseline_error": eb,
                    "candidate_error": ec, "margin": margin, "trials": trials,
                    "seed": seed, "wald_false_ship": wald / trials,
                    "score_false_ship": score / trials})
    return out


def table(res):
    print("| cases | scenario | right call | overall accuracy ships | "
          "no slice worse ships | gate ships | gate holds | gate asks for data |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res:
        print("| %d | %s | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            r["cases"], r["scenario"],
            "ship" if r["should_ship"] else "do not ship",
            100 * r["naive_overall_ship"], 100 * r["naive_slice_ship"],
            100 * r["gate_ship"], 100 * r["gate_hold"],
            100 * r["gate_more_data"]))


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cal = calibrate()
    print("false ship rate for one slice exactly at the margin (target <= 5%)\n")
    print("| slice cases | baseline error | candidate error | margin | "
          "v1 Wald bound | score bound |")
    print("|---|---|---|---|---|---|")
    for r in cal:
        print("| %d | %.0f%% | %.0f%% | %.0f pts | %.2f%% | %.2f%% |" % (
            r["slice_cases"], 100 * r["baseline_error"],
            100 * r["candidate_error"], 100 * r["margin"],
            100 * r["wald_false_ship"], 100 * r["score_false_ship"]))
    sys.stdout.flush()
    dev = run("dev", trials, (4800, 19200), seed=7)
    print("\ndev setting (seed 7)\n")
    table(dev)
    sys.stdout.flush()
    held = run("heldout", trials, (19200, 76800), seed=20261123)
    print("\nheld-out setting (seed 20261123)\n")
    table(held)
    with open("bench_results.json", "w") as f:
        json.dump({"note": "all data synthetic", "calibration": cal,
                   "dev": dev, "heldout": held}, f, indent=1)

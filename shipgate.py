"""shipgate: decide whether a new moderation model can replace the current one.

Input is a labeled eval set where both models have made a decision on every
case. The output is SHIP, HOLD or MORE_DATA, with the slice that caused it.
This module is the statistics. skformat.py builds the input rows from webhook
result events and reviewer ground truth decisions.

A slice is one (policy, class) pair:
  benign cases    -> the error is a false positive (a good user gets actioned)
  violating cases -> the error is a miss (bad content stays up)

Standard library only.
"""
import json
import math
import sys
from collections import namedtuple

Z95 = 1.6448536269514722  # one-sided 95%

Slice = namedtuple("Slice", "policy kind n base_err cand_err base_only cand_only")
SliceResult = namedtuple(
    "SliceResult",
    "slice base_rate cand_rate diff upper p_regress status more_cases")
Verdict = namedtuple("Verdict", "decision reasons results")

DEFAULT_MARGINS = {"false_positive": 0.02, "miss": 0.05}


def load(path):
    """Read JSONL rows: {"policy", "label", "baseline", "candidate"} (booleans
    mean "violating" / "flagged")."""
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in ("policy", "label", "baseline", "candidate"):
                if k not in r:
                    raise ValueError("line %d: missing %r" % (ln, k))
            rows.append(r)
    return rows


def tabulate(rows):
    """Collapse rows into one paired error table per (policy, class)."""
    acc = {}
    for r in rows:
        label = bool(r["label"])
        key = (r["policy"], "miss" if label else "false_positive")
        t = acc.setdefault(key, [0, 0, 0, 0, 0])
        b_wrong = bool(r["baseline"]) != label
        c_wrong = bool(r["candidate"]) != label
        t[0] += 1
        t[1] += b_wrong
        t[2] += c_wrong
        t[3] += b_wrong and not c_wrong
        t[4] += c_wrong and not b_wrong
    return [Slice(k[0], k[1], *v) for k, v in sorted(acc.items())]


def binom_tail(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2). Exact.

    For k above n/2 the terms shrink from the first one, so the sum starts at
    a log-space term and walks up with the ratio (n - i) / (i + 1). For k at
    or below n/2 it uses the symmetry P(X >= k) = 1 - P(X >= n - k + 1), which
    avoids starting from a term that underflows.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if 2 * k <= n:
        return max(0.0, 1.0 - binom_tail(n - k + 1, n))
    term = math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                    - math.lgamma(n - k + 1) - n * math.log(2))
    total = 0.0
    for i in range(k, n + 1):
        total += term
        term *= (n - i) / (i + 1.0)
        if term < total * 1e-17:
            break
    return min(1.0, total)


def wald_upper(s, z=Z95):
    """The bound version 1 of this tool used: Agresti-Min adjusted Wald
    interval (add 0.5 to each discordant count and 2 to n). Kept only so
    bench.py can measure it against the score bound; the gate does not use
    it, because it ships too often on small slices."""
    n = s.n + 2.0
    p_c = (s.cand_only + 0.5) / n
    p_b = (s.base_only + 0.5) / n
    d = p_c - p_b
    se = math.sqrt(max(p_c + p_b - d * d, 0.0) / n)
    return d + z * se, se


def score_stat(cand_only, base_only, n, delta):
    """Tango's score statistic for H0: (candidate error rate - baseline error
    rate) = delta, on paired decisions. The variance uses the maximum
    likelihood estimate of the discordant rate under H0 instead of the
    observed one, which is what keeps it honest when counts are small.
    At delta = 0 it reduces to the McNemar statistic."""
    b, c = float(cand_only), float(base_only)
    num = b - c - n * delta
    a2 = 2.0 * n
    bb = -b - c + (2.0 * n - b + c) * delta
    cc = -c * delta * (1.0 - delta)
    q = (math.sqrt(max(bb * bb - 4.0 * a2 * cc, 0.0)) - bb) / (2.0 * a2)
    var = n * (2.0 * q + delta * (1.0 - delta))
    if var <= 1e-12:
        return 0.0 if abs(num) < 1e-12 else math.copysign(float("inf"), num)
    return num / math.sqrt(var)


def paired_upper(s, z=Z95):
    """One-sided upper confidence bound on (candidate error rate - baseline
    error rate): the delta at which the score statistic equals -z, found by
    bisection. Returns (upper, se) where se = (upper - observed diff) / z is
    only used for the sample size estimate."""
    d = (s.cand_only - s.base_only) / float(s.n)
    lo, hi = d, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if score_stat(s.cand_only, s.base_only, s.n, mid) > -z:
            lo = mid
        else:
            hi = mid
    return hi, (hi - d) / z


def judge_slice(s, margin, alpha):
    diff = (s.cand_err - s.base_err) / s.n
    upper, se = paired_upper(s)
    # exact McNemar, one-sided: is the candidate worse on the cases where
    # the two models disagree?
    p = binom_tail(s.cand_only, s.cand_only + s.base_only)
    more = None
    if p < alpha:
        status = "regressed"
    elif upper < margin:
        status = "ok"
    else:
        status = "unproven"
        d_adj = upper - Z95 * se
        if d_adj < margin and se > 0:
            # the distance from the observed difference to the bound shrinks
            # roughly with 1/sqrt(n)
            scale = (Z95 * se / (margin - d_adj)) ** 2
            more = max(0, int(math.ceil(s.n * scale)) - s.n)
    return SliceResult(s, s.base_err / s.n, s.cand_err / s.n, diff, upper, p,
                       status, more)


def evaluate(rows, margins=None, alpha=0.05):
    """Return a Verdict.

    HOLD       some slice is significantly worse (exact McNemar, Bonferroni
               corrected across slices so more policies do not mean more
               false alarms).
    SHIP       no slice regressed and every slice's upper bound on the error
               increase is inside its margin.
    MORE_DATA  nothing regressed but at least one slice is too small to rule
               out a regression as large as the margin.
    """
    margins = dict(DEFAULT_MARGINS, **(margins or {}))
    slices = tabulate(rows)
    if not slices:
        raise ValueError("no cases")
    per_slice_alpha = alpha / len(slices)
    results = [judge_slice(s, margins[s.kind], per_slice_alpha) for s in slices]
    bad = [r for r in results if r.status == "regressed"]
    open_ = [r for r in results if r.status == "unproven"]
    if bad:
        return Verdict("HOLD", bad, results)
    if open_:
        return Verdict("MORE_DATA", open_, results)
    return Verdict("SHIP", [], results)


def report(v):
    out = ["decision: %s" % v.decision, ""]
    out.append("%-14s %-14s %6s %8s %8s %8s %8s  %s" % (
        "policy", "error", "n", "base", "cand", "upper", "p", "status"))
    for r in v.results:
        s = r.slice
        note = r.status
        if r.status == "unproven" and r.more_cases is not None:
            note += " (about %d more cases)" % r.more_cases
        out.append("%-14s %-14s %6d %7.1f%% %7.1f%% %+7.1f%% %8.4f  %s" % (
            s.policy, s.kind, s.n, 100 * r.base_rate, 100 * r.cand_rate,
            100 * r.upper, r.p_regress, note))
    return "\n".join(out)


USAGE = """usage:
  python shipgate.py decisions.jsonl
  python shipgate.py --baseline B.jsonl --candidate C.jsonl --truth T.jsonl
                     [--config gate_config.json]"""

EXIT = {"SHIP": 0, "HOLD": 1, "MORE_DATA": 3}


def main(argv):
    args = argv[1:]
    if len(args) == 1 and not args[0].startswith("-"):
        v = evaluate(load(args[0]))
        print(report(v))
        return EXIT[v.decision]
    opts = {}
    while len(args) >= 2 and args[0] in ("--baseline", "--candidate",
                                         "--truth", "--config"):
        opts[args[0][2:]] = args[1]
        args = args[2:]
    if args or not all(k in opts for k in ("baseline", "candidate", "truth")):
        print(USAGE)
        return 2
    import skformat
    try:
        rv = skformat.gate_files(opts["baseline"], opts["candidate"],
                                 opts["truth"], opts.get("config"))
    except (ValueError, OSError) as e:
        print("error: %s" % e)
        return 2
    print(skformat.report_run(rv))
    return EXIT[rv.decision]


if __name__ == "__main__":
    sys.exit(main(sys.argv))

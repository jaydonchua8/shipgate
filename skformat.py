"""skformat: build shipgate rows from the two record shapes in SafetyKit's
public API docs.

  result events   one JSON object per line, shaped like the documented
                  `workflow.succeeded` / `workflow.failed` webhook payloads:
                  {"type", "namespace", "id", "request_id", "processed_at",
                   "output": {"actions": [...], "labels": [{"label": ...}]}}
                  One file per model run (baseline, candidate).

  ground truth    one JSON object per line, shaped like the documented
                  POST /v1/agent_decisions request body:
                  {"reference_id", "content_type", "policy_id", "decision",
                   "reviewer_id"?, "agent_type"?, "action"?, "comment"?}

The docs leave label ids, policy ids and verdict values to each account, so a
config file maps labels to policies and verdicts to violating / benign.

Nothing here calls the API. It only reads files in those shapes.
Standard library only.
"""
import json
from collections import namedtuple
from datetime import datetime, timezone

import shipgate

DEFAULT_CONFIG = {
    "label_to_policy": {},
    "violating_decisions": ["violation"],
    "benign_decisions": ["no_violation"],
    "agent_type_priority": [],
    "min_candidate_coverage": 0.98,
    "margins": dict(shipgate.DEFAULT_MARGINS),
    "alpha": 0.05,
}

RunVerdict = namedtuple("RunVerdict", "decision verdict stats notes")


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            user = json.load(f)
        unknown = [k for k in user if k not in cfg and not k.startswith("_")]
        if unknown:
            raise ValueError("config: unknown keys %s" % sorted(unknown))
        cfg.update({k: v for k, v in user.items() if not k.startswith("_")})
    both = set(cfg["violating_decisions"]) & set(cfg["benign_decisions"])
    if both:
        raise ValueError("config: verdicts listed as both violating and "
                         "benign: %s" % sorted(both))
    if not 0.0 <= cfg["min_candidate_coverage"] <= 1.0:
        raise ValueError("config: min_candidate_coverage must be in [0, 1]")
    return cfg


def read_jsonl(path):
    out = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError("%s line %d: not JSON (%s)" % (path, ln, e))
            if not isinstance(obj, dict):
                raise ValueError("%s line %d: expected an object" % (path, ln))
            out.append(obj)
    return out


def _when(ev):
    """processed_at as a sortable value. Missing or unparseable sorts first,
    so a dated delivery always beats an undated one."""
    t = ev.get("processed_at")
    if not isinstance(t, str):
        return datetime.min
    try:
        d = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return d


def index_events(events, cfg, name="run"):
    """Collapse one run's events to {object id: set of flagged policies or
    None}. None means the last word on that object was workflow.failed.

    Returns (index, stats). Rules:
      - the same object can be delivered more than once; the delivery with
        the latest processed_at wins, and a later line wins a tie
      - a succeeded delivery beats a failed one for the same object, since
        workflow.failed carries no processed_at to order them by
      - labels that are not in label_to_policy are ignored and counted
      - event types other than the two workflow ones are ignored and counted
    """
    stats = {"events": 0, "objects": 0, "duplicates": 0, "failed": 0,
             "other_types": 0, "unmapped_labels": {}}
    best = {}
    for i, ev in enumerate(events):
        stats["events"] += 1
        typ = ev.get("type")
        if typ not in ("workflow.succeeded", "workflow.failed"):
            stats["other_types"] += 1
            continue
        oid = ev.get("id")
        if not isinstance(oid, str) or not oid:
            raise ValueError("%s event %d: missing id" % (name, i + 1))
        if typ == "workflow.failed":
            if oid in best:
                stats["duplicates"] += 1
            else:
                best[oid] = (None, None)
            continue
        out = ev.get("output")
        if not isinstance(out, dict) or not isinstance(out.get("labels"), list):
            raise ValueError("%s event %d (%s): workflow.succeeded without "
                             "output.labels" % (name, i + 1, oid))
        flagged = set()
        for lab in out["labels"]:
            lid = lab.get("label") if isinstance(lab, dict) else None
            if not isinstance(lid, str):
                raise ValueError("%s event %d (%s): label entry without a "
                                 "label id" % (name, i + 1, oid))
            pol = cfg["label_to_policy"].get(lid)
            if pol is None:
                stats["unmapped_labels"][lid] = \
                    stats["unmapped_labels"].get(lid, 0) + 1
            else:
                flagged.add(pol)
        if oid in best:
            stats["duplicates"] += 1
            prev = best[oid][0]
            # times are only parsed when two deliveries have to be ordered
            if prev is not None and _when(prev) > _when(ev):
                continue
        best[oid] = (ev, flagged)
    index = {oid: v[1] for oid, v in best.items()}
    stats["objects"] = len(index)
    stats["failed"] = sum(1 for v in index.values() if v is None)
    return index, stats


def resolve_truth(decisions, cfg):
    """Collapse reviewer decisions to {(reference_id, policy_id): bool}.

    The docs say to send one request per object and policy, and to send the
    final corrected decision. Real feeds still carry several decisions for
    one pair (a vendor review, then an appeal). Rules:
      - only decisions from the highest ranked agent_type present for that
        pair count (agent_type_priority, first is highest; unlisted and
        missing agent types rank last)
      - if those decisions disagree, the pair is contested and left out
      - a verdict that is not in the config is an error, because guessing
        what a verdict means would silently corrupt the eval
    """
    prio = {a: i for i, a in enumerate(cfg["agent_type_priority"])}
    last = len(prio)
    viol = set(cfg["violating_decisions"])
    ben = set(cfg["benign_decisions"])
    known = set(cfg["label_to_policy"].values())
    stats = {"decisions": 0, "pairs": 0, "contested": 0, "overridden": 0,
             "unknown_policy": 0, "contested_by_policy": {}}
    groups = {}
    for i, d in enumerate(decisions):
        stats["decisions"] += 1
        for k in ("reference_id", "content_type", "policy_id", "decision"):
            if not isinstance(d.get(k), str) or not d[k]:
                raise ValueError("decision %d: missing required field %r"
                                 % (i + 1, k))
        if d["decision"] in viol:
            label = True
        elif d["decision"] in ben:
            label = False
        else:
            raise ValueError("decision %d: verdict %r is not in "
                             "violating_decisions or benign_decisions"
                             % (i + 1, d["decision"]))
        if known and d["policy_id"] not in known:
            stats["unknown_policy"] += 1
            continue
        rank = prio.get(d.get("agent_type"), last)
        groups.setdefault((d["reference_id"], d["policy_id"]), []).append(
            (rank, label))
    truth = {}
    for key, votes in groups.items():
        top = min(r for r, _ in votes)
        labels = set(l for r, l in votes if r == top)
        if len(labels) > 1:
            stats["contested"] += 1
            c = stats["contested_by_policy"]
            c[key[1]] = c.get(key[1], 0) + 1
            continue
        label = labels.pop()
        if any(l != label for r, l in votes if r != top):
            stats["overridden"] += 1
        truth[key] = label
    stats["pairs"] = len(truth)
    return truth, stats


def join(baseline_events, candidate_events, decisions, cfg):
    """Return (rows, stats). rows are shipgate rows; a pair is only a row if
    both runs produced a result for the object."""
    base, bs = index_events(baseline_events, cfg, "baseline")
    cand, cs = index_events(candidate_events, cfg, "candidate")
    truth, ts = resolve_truth(decisions, cfg)
    rows = []
    miss = {"baseline_missing": 0, "candidate_missing": 0, "both_missing": 0}
    for (oid, pol), label in truth.items():
        b, c = base.get(oid), cand.get(oid)
        if b is None and c is None:
            miss["both_missing"] += 1
        elif b is None:
            miss["baseline_missing"] += 1
        elif c is None:
            miss["candidate_missing"] += 1
        else:
            rows.append({"policy": pol, "label": label, "id": oid,
                         "baseline": pol in b, "candidate": pol in c})
    n = len(truth)
    with_base = n - miss["baseline_missing"] - miss["both_missing"]
    stats = {"baseline": bs, "candidate": cs, "truth": ts, "rows": len(rows),
             "candidate_coverage": (len(rows) / float(with_base)
                                    if with_base else 0.0)}
    stats.update(miss)
    return rows, stats


def gate(baseline_events, candidate_events, decisions, cfg=None):
    """Join, check that the candidate actually answered, then run the gate.

    Coverage is checked first. A candidate that fails on labeled cases the
    baseline handled would otherwise look fine, because the cases it could
    not process drop out of the comparison.
    """
    cfg = cfg or dict(DEFAULT_CONFIG)
    rows, stats = join(baseline_events, candidate_events, decisions, cfg)
    notes = []
    if stats["truth"]["contested"]:
        notes.append("%d contested ground truth pairs left out"
                     % stats["truth"]["contested"])
    if stats["candidate"]["unmapped_labels"]:
        notes.append("labels not in label_to_policy were ignored: %s"
                     % ", ".join(sorted(stats["candidate"]["unmapped_labels"])))
    if not rows:
        raise ValueError("no labeled case has a result from both runs; check "
                         "that reference_id matches the ingested id and that "
                         "label_to_policy covers the policy ids")
    verdict = shipgate.evaluate(rows, cfg["margins"], cfg["alpha"])
    decision = verdict.decision
    if stats["candidate_coverage"] < cfg["min_candidate_coverage"]:
        decision = "HOLD"
        notes.insert(0, "candidate has no result for %d of %d labeled cases "
                     "the baseline handled (coverage %.1f%%, minimum %.1f%%)"
                     % (stats["candidate_missing"],
                        stats["candidate_missing"] + stats["rows"],
                        100 * stats["candidate_coverage"],
                        100 * cfg["min_candidate_coverage"]))
    return RunVerdict(decision, verdict, stats, notes)


def gate_files(baseline_path, candidate_path, truth_path, config_path=None):
    cfg = load_config(config_path)
    return gate(read_jsonl(baseline_path), read_jsonl(candidate_path),
                read_jsonl(truth_path), cfg)


def report_run(rv):
    s = rv.stats
    lines = [
        "baseline run   %6d events  %6d objects  %4d duplicate deliveries  "
        "%4d failed" % (s["baseline"]["events"], s["baseline"]["objects"],
                        s["baseline"]["duplicates"], s["baseline"]["failed"]),
        "candidate run  %6d events  %6d objects  %4d duplicate deliveries  "
        "%4d failed" % (s["candidate"]["events"], s["candidate"]["objects"],
                        s["candidate"]["duplicates"], s["candidate"]["failed"]),
        "ground truth   %6d decisions -> %d usable pairs (%d contested, "
        "%d settled by a higher ranked review)" % (
            s["truth"]["decisions"], s["truth"]["pairs"],
            s["truth"]["contested"], s["truth"]["overridden"]),
        "joined         %6d cases, candidate coverage %.1f%%" % (
            s["rows"], 100 * s["candidate_coverage"]),
    ]
    for n in rv.notes:
        lines.append("note: " + n)
    body = shipgate.report(rv.verdict)
    if rv.decision != rv.verdict.decision:
        body = body.replace("decision: %s" % rv.verdict.decision,
                            "decision: %s (coverage; slice statistics alone "
                            "say %s)" % (rv.decision, rv.verdict.decision), 1)
    return "\n".join(lines) + "\n\n" + body

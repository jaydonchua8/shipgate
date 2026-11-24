"""synth: SYNTHETIC eval data in the record shapes skformat reads.

Everything this module produces is made up. Policy names, label ids, error
rates and reviewer behaviour are my own choices, not SafetyKit data.

Two generator settings exist on purpose:
  DEV       balanced policies, clean labels, no delivery problems. I picked
            the gate's default margins and coverage floor while looking at
            results from this one.
  HELDOUT   uneven policy sizes, different base rates and class balance,
            noisy vendor reviews with partial expert re-review, duplicate
            and failed deliveries. Used only to report results.
"""
import random

LABEL = {"counterfeit": "policy.counterfeit", "dropship": "policy.dropship",
         "hazmat": "policy.hazmat", "ip": "policy.ip_violation",
         "recalled": "policy.recalled_product", "adult": "policy.adult_content"}
POLICIES = sorted(LABEL)
NOISE_LABEL = "quality.low_resolution_image"   # not a policy; must be ignored

DEV = {
    "name": "dev",
    "weights": {p: 1.0 for p in POLICIES},
    "violating_share": 0.20,
    "base": {p: {"false_positive": 0.03, "miss": 0.15} for p in POLICIES},
    "rho": 0.6,              # share of cases both models find equally hard
    "review_noise": 0.0,     # chance a vendor review is wrong
    "second_vendor": 0.0,    # chance a second vendor review exists
    "expert_share": 0.0,     # chance an expert re-review exists (always right)
    "dup_rate": 0.0,         # chance an event is delivered twice
    "fail_rate": 0.0,        # chance a run fails on an object
}

HELDOUT = {
    "name": "heldout",
    "weights": {"counterfeit": 5.0, "dropship": 3.0, "adult": 2.0,
                "ip": 1.5, "hazmat": 1.0, "recalled": 0.6},
    "violating_share": 0.30,
    "base": {"counterfeit": {"false_positive": 0.04, "miss": 0.12},
             "dropship": {"false_positive": 0.05, "miss": 0.20},
             "adult": {"false_positive": 0.01, "miss": 0.06},
             "ip": {"false_positive": 0.03, "miss": 0.18},
             "hazmat": {"false_positive": 0.02, "miss": 0.10},
             "recalled": {"false_positive": 0.02, "miss": 0.25}},
    "rho": 0.4,
    "review_noise": 0.04,
    "second_vendor": 0.15,
    "expert_share": 0.25,
    "dup_rate": 0.05,
    "fail_rate": 0.01,
}


def scenarios(settings):
    """name -> (candidate error rates per policy, is shipping the right call).

    Rates are relative to the setting's baseline. The regressions are placed
    on the smallest-weight policy so the held-out set tests the hard case.
    """
    base = settings["base"]
    better = {p: {"false_positive": base[p]["false_positive"] * 2 / 3.0,
                  "miss": base[p]["miss"] * 2 / 3.0} for p in POLICIES}
    target = min(POLICIES, key=lambda p: (settings["weights"][p], p))

    def with_change(kind, add):
        c = {p: dict(v) for p, v in better.items()}
        c[target][kind] = base[target][kind] + add
        return c

    return {
        "better_everywhere": (better, True),
        "no_change": ({p: dict(v) for p, v in base.items()}, True),
        "fp_regression_large": (with_change("false_positive", 0.06), False),
        "miss_regression_large": (with_change("miss", 0.15), False),
        # exactly at the default margins: the smallest change the gate
        # promises not to ship
        "fp_regression_at_margin": (with_change("false_positive", 0.02), False),
        "miss_regression_at_margin": (with_change("miss", 0.05), False),
    }


def _event(oid, flagged_policy, rng, seq, minute):
    labels = []
    if flagged_policy:
        labels.append({"label": LABEL[flagged_policy],
                       "explanation": "synthetic"})
    if rng.random() < 0.03:
        labels.append({"label": NOISE_LABEL})
    return {"type": "workflow.succeeded", "namespace": "products", "id": oid,
            "request_id": "req_%08d" % seq,
            "processed_at": "2026-11-01T%02d:%02d:%02d.000Z" % (
                10 + minute // 3600, minute // 60 % 60, minute % 60),
            "output": {"actions": [], "labels": labels}}


def _emit(events, oid, flagged, settings, rng, seq):
    if rng.random() < settings["fail_rate"]:
        events.append({"type": "workflow.failed", "namespace": "products",
                       "id": oid, "request_id": "req_%08d" % seq,
                       "error": "synthetic failure"})
        return
    t = rng.randrange(0, 3 * 3600)
    events.append(_event(oid, flagged, rng, seq, t))
    if rng.random() < settings["dup_rate"]:
        # redelivery of the same result, sometimes arriving out of order
        events.append(dict(events[-1]))
        j = rng.randrange(0, len(events))
        events[j], events[-1] = events[-1], events[j]


def generate(settings, cand_rates, n, rng):
    """Return (baseline_events, candidate_events, decisions, clean_rows).

    clean_rows uses the true labels and both models' real decisions, before
    review noise or delivery problems. Tests use it to check the join.
    """
    pols = POLICIES
    w = [settings["weights"][p] for p in pols]
    total = sum(w)
    base_ev, cand_ev, decisions, clean = [], [], [], []
    for i in range(n):
        x, policy = rng.random() * total, pols[-1]
        for p, wp in zip(pols, w):
            if x < wp:
                policy = p
                break
            x -= wp
        oid = "listing_%07d" % i
        label = rng.random() < settings["violating_share"]
        kind = "miss" if label else "false_positive"
        eb, ec = settings["base"][policy][kind], cand_rates[policy][kind]
        u = rng.random()
        b_wrong = u < eb
        c_wrong = (u < ec) if rng.random() < settings["rho"] \
            else (rng.random() < ec)
        b_flag, c_flag = label != b_wrong, label != c_wrong
        clean.append({"policy": policy, "label": label, "id": oid,
                      "baseline": b_flag, "candidate": c_flag})
        _emit(base_ev, oid, policy if b_flag else None, settings, rng, i)
        _emit(cand_ev, oid, policy if c_flag else None, settings, rng, n + i)

        def review(agent, reviewer, noise):
            seen = label != (rng.random() < noise)
            decisions.append({
                "reference_id": oid, "content_type": "product_listing",
                "policy_id": policy,
                "decision": "violation" if seen else "no_violation",
                "agent_type": agent, "reviewer_id": reviewer})

        review("bpo_review", "reviewer_%02d" % rng.randrange(20),
               settings["review_noise"])
        if rng.random() < settings["second_vendor"]:
            review("bpo_review", "reviewer_%02d" % rng.randrange(20),
                   settings["review_noise"])
        if rng.random() < settings["expert_share"]:
            review("policy_expert", "expert_%02d" % rng.randrange(4), 0.0)
    return base_ev, cand_ev, decisions, clean

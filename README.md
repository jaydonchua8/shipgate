# shipgate

Decide whether a new moderation model can replace the current one, from two runs of result events and a file of reviewer ground truth decisions. The answer is `SHIP`, `HOLD` or `MORE_DATA`, with the policy that caused it.

This is an independent prototype. All data in this repository is synthetic. It is not affiliated with, reviewed by or connected to SafetyKit, and it does not call their API.

## The problem

A moderation model is checked against many policies at once. When a new model comes out, an overall accuracy number can go up while one small policy starts flagging many more legitimate users, because the improved policies outweigh it. The question I wanted to answer is what "the evals look good" should mean per policy, and how to answer it from the records a platform already has instead of a hand-built table.

Version 1 of this project only did the statistics, on a flat file I invented (`policy, label, baseline, candidate`). It skipped everything that happens before that table exists: duplicate result deliveries, objects a model run failed on, and reviewers who disagree with each other. It also used a confidence bound that I had not checked.

## Why it matters to SafetyKit

What I verified from public pages:

- OpenAI's case study on SafetyKit says the team benchmarks each new model against its hardest cases and often deploys the best one the same day, and that it routes content to a separate agent per violation category. (https://openai.com/index/safetykit)
- SafetyKit's docs define a `workflow.succeeded` webhook event with `id`, `processed_at` and `output.labels` (a list of `{"label": ...}`), and a `workflow.failed` event for objects that could not be processed. (https://docs.safetykit.com/webhooks/event-types/workflow-succeeded/)
- The docs define Agent Decisions: a customer sends the correct outcome for one object and one policy (`reference_id`, `content_type`, `policy_id`, `decision`, optional `reviewer_id`, `agent_type`), and SafetyKit uses these as ground truth to measure review quality. (https://docs.safetykit.com/agent-decisions/)
- The docs say content types, policy ids and verdict values are configured per account.

So the two record shapes this tool reads are the ones those docs describe: result events for a model's output, and agent decisions for ground truth. How SafetyKit actually runs its evals internally is not public and I have not guessed at it.

## What it does

```
python shipgate.py --baseline B.jsonl --candidate C.jsonl --truth T.jsonl --config gate_config.json
```

- `B.jsonl` and `C.jsonl` hold one result event per line for the current model and the candidate, run on the same objects.
- `T.jsonl` holds one agent decision per line.
- `gate_config.json` maps label ids to policy ids and verdict values to violating or benign, and ranks `agent_type` values.

It joins the three files into one row per (object, policy), then judges every (policy, error type) slice:

- `HOLD`: the candidate is measurably worse on a slice, or it returned no result for too many labeled cases.
- `SHIP`: nothing is worse, and every slice has enough cases to rule out a regression larger than its margin (2 points for false positives, 5 points for misses by default).
- `MORE_DATA`: nothing is measurably worse, but a slice is too small to tell. It estimates how many more labeled cases that slice needs.

Exit codes are 0, 1 and 3 for those answers and 2 for bad input, so it can run as a CI step. The old flat format still works: `python shipgate.py sample_decisions.jsonl`.

## Architecture

| file | role |
|---|---|
| `skformat.py` | Reads result events and agent decisions, resolves duplicates, failures and reviewer conflicts, joins them into rows, checks candidate coverage. |
| `shipgate.py` | The statistics and the command line. Paired per-slice tests on the joined rows. |
| `synth.py` | Synthetic data in the same record shapes, with a dev setting and a harder held-out setting. |
| `bench.py` | Simulates candidates of known quality through the whole path and counts right and wrong calls. |
| `demo.py` | One end to end example in a few seconds. |
| `gate_config.json` | Example account mapping (synthetic policy and label names). |
| `test_shipgate.py`, `test_skformat.py` | 53 tests. |

## Key decisions and tradeoffs

- **Latest delivery wins.** If the same object id appears more than once in a run, the event with the latest `processed_at` is used, whatever order the lines are in. A succeeded event beats a failed one for the same object, because the failed payload has no timestamp to order them by.
- **Coverage is checked before accuracy.** A candidate that fails on labeled cases the baseline handled would otherwise look fine, since those cases drop out of a paired comparison. If the candidate covers less than 98% of them the answer is `HOLD`, whatever the slices say. Baseline failures do not count against the candidate.
- **Conflicting reviews are not averaged.** For one (object, policy), only decisions from the highest ranked `agent_type` count. If those disagree, the pair is left out and counted. I chose to lose a few cases over inventing a majority rule for two reviewers.
- **Unknown verdicts are errors.** A verdict that is not in the config stops the run. Labels that are not in the config are ignored and listed, because an account can have labels that are not under evaluation.
- **Exact test for regressions.** One-sided exact McNemar test on the cases where the two models disagree, with the threshold divided by the number of slices (Bonferroni) so adding policies does not add false holds.
- **Score bound for shipping.** Version 1 used an adjusted Wald interval for the upper bound on the error increase. When I measured it on a slice whose true regression was exactly the margin, it cleared the margin 4.67% to 6.50% of the time against a 5% target. I replaced it with Tango's score bound, which measured 4.20% to 5.22% on the same cases (table below). The change is small, but the old bound was above target in 6 of 7 cases and the new one in 3 of 7, by at most 0.22 points.
- **No correction on the ship test.** Every slice has to pass, so a candidate with one slice at the margin can only ship if that slice passes. The 5% applies per slice without a multiple testing correction.

## Install and usage

Python 3.8 or newer. Nothing to install.

```
python -m unittest          # 53 tests, about 10 seconds
python demo.py              # a few seconds
python shipgate.py sample_decisions.jsonl
python bench.py             # about 25 minutes on 2 cores; python bench.py 10 for a quick look
```

## Example

`python demo.py` generates 40,000 synthetic objects with the held-out settings (4% vendor review error, duplicate and failed deliveries), where the candidate is better on five policies and has 6 points more false positives on the smallest one. Output, with the two example records shortened:

```
overall accuracy: baseline 91.33%, candidate 93.13% -> an overall accuracy rule says SHIP

baseline run    41983 events   40000 objects  1983 duplicate deliveries   436 failed
candidate run   42003 events   40000 objects  2003 duplicate deliveries   358 failed
ground truth    55993 decisions -> 39628 usable pairs (372 contested, 466 settled by a higher ranked review)
joined          38841 cases, candidate coverage 99.1%
note: 372 contested ground truth pairs left out
note: labels not in label_to_policy were ignored: quality.low_resolution_image

decision: HOLD

policy         error               n     base     cand    upper        p  status
adult          false_positive   4076     2.0%     1.6%    -0.0%   0.9756  ok
adult          miss             1798    12.7%     9.8%    -1.8%   1.0000  ok
counterfeit    false_positive  10330     4.7%     3.6%    -0.8%   1.0000  ok
counterfeit    miss             4613    16.8%    12.6%    -3.3%   1.0000  ok
dropship       false_positive   6260     5.2%     4.4%    -0.4%   0.9984  ok
dropship       miss             2672    24.2%    18.7%    -4.1%   1.0000  ok
hazmat         false_positive   2059     3.0%     2.6%    +0.2%   0.8942  ok
hazmat         miss              905    14.8%    10.4%    -2.9%   1.0000  ok
ip             false_positive   3049     3.8%     2.8%    -0.5%   0.9994  ok
ip             miss             1339    23.4%    17.0%    -4.5%   1.0000  ok
recalled       false_positive   1213     3.1%     9.3%    +7.6%   0.0000  regressed
recalled       miss              527    30.4%    22.8%    -4.4%   1.0000  ok
```

The measured baseline rates are higher than the rates I simulated (for example 2.0% false positives on `adult` against a true 1%), because some of the remaining ground truth is wrong. That is the label noise showing up as model error.

## Tests and verified results

`python -m unittest` runs 53 tests: the 17 from version 1, unchanged, and 36 new ones covering the documented example payload, duplicate and out of order deliveries, timezone offsets, failed runs, conflicting and overridden reviews, unknown verdicts, malformed lines, the coverage hold, the command line exit codes, and the score bound (checked against a brute force maximum likelihood search).

All numbers below are from `python bench.py` on synthetic data, 100 simulated eval sets per row, so each percentage has a standard error of up to 5 points.

I chose the default margins and the coverage floor while looking at the dev setting (6 equal policies, clean labels). The held-out setting uses a different seed and different generator settings: policy sizes from about 5% to 38% of cases, different error rates per policy, 30% violating instead of 20%, 4% vendor review error with 25% expert re-review, 5% duplicate deliveries and 1% failed objects. The regressions are placed on the smallest policy.

Confidence bound alone, one slice exactly at the margin, 4,000 trials per row (target 5% or less):

| slice cases | baseline error | candidate error | margin | v1 Wald bound | score bound |
|---|---|---|---|---|---|
| 40 | 15% | 20% | 5 pts | 4.67% | 4.55% |
| 160 | 15% | 20% | 5 pts | 5.90% | 5.08% |
| 640 | 15% | 20% | 5 pts | 5.17% | 4.80% |
| 264 | 25% | 30% | 5 pts | 5.42% | 5.05% |
| 160 | 3% | 5% | 2 pts | 6.50% | 4.20% |
| 640 | 3% | 5% | 2 pts | 5.60% | 4.47% |
| 2560 | 3% | 5% | 2 pts | 5.65% | 5.22% |

Dev setting (seed 7):

| cases | scenario | right call | overall accuracy ships | no slice worse ships | gate ships | gate holds | gate asks for data |
|---|---|---|---|---|---|---|---|
| 4800 | better_everywhere | ship | 100.0% | 69.0% | 85.0% | 0.0% | 15.0% |
| 4800 | no_change | ship | 50.0% | 0.0% | 6.0% | 1.0% | 93.0% |
| 4800 | fp_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 4800 | miss_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 85.0% | 15.0% |
| 4800 | fp_regression_at_margin | do not ship | 100.0% | 0.0% | 1.0% | 39.0% | 60.0% |
| 4800 | miss_regression_at_margin | do not ship | 100.0% | 6.0% | 9.0% | 7.0% | 84.0% |
| 19200 | better_everywhere | ship | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| 19200 | no_change | ship | 49.0% | 0.0% | 89.0% | 3.0% | 8.0% |
| 19200 | fp_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 19200 | miss_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 19200 | fp_regression_at_margin | do not ship | 100.0% | 0.0% | 1.0% | 99.0% | 0.0% |
| 19200 | miss_regression_at_margin | do not ship | 100.0% | 0.0% | 3.0% | 68.0% | 29.0% |

Held-out setting (seed 20261123):

| cases | scenario | right call | overall accuracy ships | no slice worse ships | gate ships | gate holds | gate asks for data |
|---|---|---|---|---|---|---|---|
| 19200 | better_everywhere | ship | 100.0% | 68.0% | 96.0% | 0.0% | 4.0% |
| 19200 | no_change | ship | 54.0% | 0.0% | 40.0% | 3.0% | 57.0% |
| 19200 | fp_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 98.0% | 2.0% |
| 19200 | miss_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 95.0% | 5.0% |
| 19200 | fp_regression_at_margin | do not ship | 100.0% | 0.0% | 0.0% | 27.0% | 73.0% |
| 19200 | miss_regression_at_margin | do not ship | 100.0% | 8.0% | 9.0% | 10.0% | 81.0% |
| 76800 | better_everywhere | ship | 100.0% | 94.0% | 100.0% | 0.0% | 0.0% |
| 76800 | no_change | ship | 51.0% | 0.0% | 90.0% | 7.0% | 3.0% |
| 76800 | fp_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 76800 | miss_regression_large | do not ship | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 76800 | fp_regression_at_margin | do not ship | 100.0% | 0.0% | 2.0% | 98.0% | 0.0% |
| 76800 | miss_regression_at_margin | do not ship | 100.0% | 0.0% | 6.0% | 60.0% | 34.0% |

What I take from this:

- The gate shipped none of the 800 large-regression eval sets (400 dev, 400 held-out). The overall accuracy rule shipped all 800.
- On held-out data with 19,200 cases it held 98% of large false positive regressions and 95% of large miss regressions on a policy that is under 5% of the eval set.
- The weak spot is a miss regression exactly at the margin. The gate shipped it in 9% of held-out runs at 19,200 cases and 6% at 76,800, above the 5% target. At 100 runs I cannot separate that from noise, but I expect part of it is real: wrong ground truth labels shrink the measured difference between two models, which pushes a regression at the margin under it. The gate does not correct for that.
- It held an unchanged model in 1% to 7% of runs. The 7% row is above the 5% target and I have not run enough trials to say whether that is noise.
- Uneven policies cost data. On the held-out setting the gate needed 76,800 cases to ship an unchanged model 90% of the time, against 19,200 on the balanced dev setting, because the smallest policy decides.

## Limitations and next steps

- Everything is synthetic, including the policy names, label ids, verdict values and error rates. I have no access to real moderation decisions.
- I have only read SafetyKit's public docs pages. Field names match what those pages show; I have not tested against the real API, and the docs I read do not say whether webhooks are redelivered, so the duplicate handling is a precaution.
- Agent decisions carry no timestamp in the documented request body, so two decisions of the same rank cannot be ordered and a disagreement is dropped instead of resolved.
- It does not model label noise. The next step I would take is estimating reviewer error from the pairs that have both a vendor review and an expert review, and adjusting the ship test to account for it.
- Simulated cases are independent. Real eval sets contain near duplicates, which would make the bounds too narrow.
- Decisions are binary per policy. Confidence values, severity and enforcement actions in the event payload are ignored.
- The margins are numbers I picked. The right ones depend on what a false positive costs a given platform.
- It is Python, and the team it was written with in mind works in TypeScript.

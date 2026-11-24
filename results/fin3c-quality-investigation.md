# `fastest` vs `fast`: quality investigation

`fastest` adds two output-changing steps to `fast`: the fused loop controller for prefill chunks (ctlf2) and split-K
small matmuls for decode (dsm). Each is as accurate as the stock kernel against an fp64 reference, but in this looped
model the changed rounding moves the output distribution (KL 0.0035 on the text corpus). The user asked for the
quality cost to be fully quantified. All tests are paired `fastest` vs `fast`, pre-registered with a decision rule
before running, with 95 % confidence intervals and the smallest effect each test can detect (MDE, 80 % power).

## Task-level results

| test | `fast` | `fastest` | difference, 95 % CI | MDE |
|---|---:|---:|---|---|
| Fresh bug-fix tasks, post-cutoff (38 Go / Python PRs, OpenCode, effort xhigh, isolated network, 40-min cap) | 12 / 38 | 11 / 38 | -1 task [-5, +2], McNemar p 1.0 | ~5 tasks |
| codeeval_long: HumanEval+ / MBPP+ plus tests, 542 tasks x 2 samples at T=1, ~2K-token context | 79.5 % | 80.7 % | +1.2 pp [-0.5, +2.9] | 2.4 pp |
| taskeval_long: 400 exact-answer problems, greedy, ~2K-token context | 365 | 369 | +1.0 pp [-2.0, +4.0], McNemar p 0.62 | ~4 pp |
| CJK risk on 177 agentic requests (expected CJK tokens per 1,000) | 0.000 | 0.000 | 0 | |
| Tool calls in the fresh runs: invalid JSON or unknown tool | 0 / 3,377 | 0 / 3,445 | both <= 0.11 % (95 % upper bound) | |
| Empty replies in the fresh runs | 0 | 0 | | |

The long-context versions of codeeval and taskeval use the unchanged tasks and scorers with a ~2K-token repository
file as context and 2 requests at a time, so that both `fastest` changes act (they only run on prefill chunks of >=
1,024 rows and decode batches of <= 16 rows, as in single-user agentic use).

## Distribution distance

| vs `fast` | KL, text corpus (teacher-forced) | NLL change, text corpus (nats / token) | KL, agentic decode path | greedy flips per 1,000 tokens |
|---|---|---|---|---|
| split-K matmuls only (dsm) | 0 | 0 | 0.0014 [0.0013, 0.0016] | 9.6 |
| fused controller only (ctlf2) | 0.0035 | +0.0022 [+0.0002, +0.0044] | 0.0019 [0.0016, 0.0022] | 10.0 |
| `fastest` (both) | 0.0035 [0.0031, 0.0040] | +0.0022 [+0.0002, +0.0044] | 0.0020 [0.0018, 0.0023] | 9.3 |
| fin-w8 (int8 step only, no far-block path) | 0.0007 | +0.0001 | 0.0036 [0.0030, 0.0043] | 14.6 |
| stock | 0.0076 | +0.0012 | 0.0039 [0.0031, 0.0048] | 15.7 |

Agentic decode path: the 177 P1 requests decoded greedily (256 tokens), compared position by position while both
token histories agree; CIs bootstrap over the 25 sessions. On this workload `fastest` is closer to `fast` than
fin-w8 or stock are, and it is as far from stock as `fast` (KL vs stock 0.0039 for both).

## Determinism

`fast` and `fastest` each on two separate boots: 177 / 177 greedy P1 outputs identical, and a 17.6K-token request
decoded to 2,048 tokens identical within and across boots.

## Reading

No task metric shows a loss. The long-context coding test excludes a loss larger than 0.5 pp. The fresh-set and
exact-answer tests only exclude large losses (about 5 tasks, about 4 pp). The one significant change is a 0.2 %
higher NLL on the text corpus, smaller than the spread of these configurations around stock.

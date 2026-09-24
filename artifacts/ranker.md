# Ranker

- Dataset: `synthetic`, variant `config`
- Token vocabulary: 945
- Fit: 261.97s, 120,142 parameters

## Head calibration (test split)

| Head | AUC | Positions | Positive rate |
| --- | --- | --- | --- |
| `click` | 0.8918 | 25,080 | 0.2441 |
| `cart` | 0.8912 | 25,080 | 0.2441 |
| `purchase` | 0.8998 | 25,080 | 0.0775 |
| `return` | 0.7700 | 1,602 | 0.1873 |

AUC 0.5 is coin-flipping. A head with `n/a` saw only one class, which for `return` usually means the return window swallowed every matured purchase — check `purchases_matured` in the sequence stats.

**AUC is not calibration.** It is invariant to any monotone transform of the scores, so these numbers say the heads *order* well and say nothing about whether their probabilities are on the probability scale. The blend multiplies them by money, so the next section is the one that decides whether that multiplication means anything.

## Calibration: are these probabilities, or just scores?

Measured on slates the shape serving builds them — retrieval's top 300 for 400 held-out users, with items already in the history excluded exactly as the request path excludes them, and the held-out future as the outcome.

**21% of these customers' future interactions (409 of 1,945) are with items already in their history, so `exclude_seen` removes them from the slate before the ranker ever scores them.** That is a property of the serving policy, not of the model, and it caps what any ranker here can be measured against. It is worth knowing separately that on this dataset *every* click, cart and purchase lands on an already-seen item — so the immediate next engagement is, under this policy, not a thing the system can be scored on at all.

| Head | Base rate | Mean predicted | Bias | ECE / base | Brier skill | Resolution | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `click` | 0.00222 | 0.00124 | 0.56x | 0.97 | -0.042 | 0.04% | under-confident by 1.8x, and uninformative |
| `click` **calibrated** | 0.00222 | 0.00193 | 0.87x | 0.23 | +0.000 | 0.04% | calibrated but uninformative |
| `cart` | 0.00222 | 0.00141 | 0.64x | 0.99 | -0.047 | 0.04% | under-confident by 1.6x, and uninformative |
| `cart` **calibrated** | 0.00222 | 0.00193 | 0.87x | 0.23 | +0.000 | 0.04% | calibrated but uninformative |
| `purchase` | 0.00097 | 0.00087 | 0.90x | 1.19 | -0.045 | 0.02% | under-confident by 1.1x, and uninformative |
| `purchase` **calibrated** | 0.00097 | 0.00088 | 0.91x | 0.30 | +0.000 | 0.02% | under-confident by 1.1x, and uninformative |
| `return` | – | – | – | – | – | – | not measured |

*Bias* is mean predicted over the base rate: 1.00x is right on average, 0.05x means the head believes the event is twenty times rarer than it is — and in an expected-value blend that is the same as dividing its business weight by twenty. *ECE / base* is the calibration error as a share of the base rate, because an absolute ECE of 0.003 is negligible for a coin flip and total for an event that happens three times in a thousand. *Brier skill* is against always predicting the base rate: **negative means the head is worse than that constant**, however good its AUC.

*Resolution* is the share of the outcome's variance the head's predictions actually explain, and it is the column to read first. Calibration cannot change it: a monotone rescaling moves where the probabilities sit, never how well they separate outcomes. So a head with resolution near zero can be made perfectly calibrated and will still be a constant wearing a probability's clothes — which is why the verdict says *calibrated but uninformative* rather than *calibrated* when that happens.

### The same heads, on the two distributions

| Head | Resolution (history) | Resolution (slate) | Brier skill (history) | Brier skill (slate) |
| --- | --- | --- | --- | --- |
| `click` | 40.73% | 0.04% | +0.408 | -0.042 |
| `cart` | 40.63% | 0.04% | +0.404 | -0.047 |
| `purchase` | 24.71% | 0.02% | +0.224 | -0.045 |
| `return` | 19.49% | – | +0.211 | – |

Left columns: *given this item is in the customer's history, which action did they take on it?* Right columns: *given this item is in a slate we are about to show, will they act on it?* The heads are trained on the first and deployed on the second. Every AUC in this report, and every AUC in the literature this design is drawn from, is measured on the left.

### What was fitted

| Head | Calibrator | Fitted on | Examples | Positives |
| --- | --- | --- | --- | --- |
| `click` | `platt` | `slate` | 120,000 | 232 |
| `cart` | `platt` | `slate` | 120,000 | 232 |
| `purchase` | `platt` | `slate` | 120,000 | 106 |
| `return` | `platt` | `teacher_forced` | 1,652 | 273 |

`identity` means no correction was installed — either too few outcomes to fit on, or the fit failed its own checks (it reversed the head's ordering, saturated, or did not improve calibration on the data it was fitted to). Declining is a result the report can print; a silently broken correction applied to every request is not.

`teacher_forced` as a source is a distribution mismatch stated rather than hidden: a return outcome exists only for a matured purchase, and a slate yields at most one of those per user, so the `return` head is normally fitted on history positions and applied to candidates. That is an extrapolation, and it is the reason the return term deserves the least trust in the blend.

Calibration is monotone per head, so it cannot change any single head's AUC. It changes the **blend**, because the blend sums heads that previously lived on incomparable scales — which is the entire point and also the only thing worth measuring afterwards.

One limitation these numbers cannot escape: the positive is the item the customer actually interacted with next, and every other candidate is labelled 0 including the ones they would have liked and never saw. So a calibrated P(purchase) of 0.004 means *4 in 1000 slates like this one had this item as the next purchase*, not *4 in 1000 customers would buy it*. Closing that gap needs exposure logs.
## Gate: BLOCKED

the ranker's MRR is 0.31x retrieval's on the same task (-0.4513 [-0.5069, -0.3972], p=0.000 over 200 shared users), so re-ranking would be a regression with extra latency; calibration could not be measured on the serving distribution for `return` (`return` is calibrated on history positions, which is a different distribution from the slate it is applied to), so the expected-value blend would be multiplying money by numbers of unknown scale

| Criterion | Measured | Threshold | Verdict |
| --- | --- | --- | --- |
| Orders at least as well as retrieval | 0.31x | >= 1.00x | **fail** |
| Probabilities on the right scale | 1.2x off (`cart`) | <= 4.0x | **fail** |

Ranker MRR 0.2039 against retrieval's 0.6552 on the same task and the same users.

The ranker is still written to the bundle directory so the next run has something to compare against, but it is **not** wired into the request path: the API serves retrieval order. `--force-ranker` overrides this.

## Can it pick the true next item out of 100?

| Signal | hit@1 | hit@10 | MRR | Median rank |
| --- | --- | --- | --- | --- |
| _random_ | 0.0100 | 0.1000 | 0.0519 | 50 |
| `click` | 0.1150 | 0.4050 | 0.2185 | 14 |
| `cart` | 0.0950 | 0.3850 | 0.2034 | 14 |
| `purchase` | 0.0850 | 0.4150 | 0.1906 | 15 |
| `return` | 0.0050 | 0.0800 | 0.0442 | 72 |
| **`blended`** | **0.1050** | **0.4000** | **0.2039** | **18** |

Each head is reported separately because the blend can destroy a signal the heads have: if a head beats random here and `blended` does not, the weights are wrong, not the model.

## Does re-ranking beat retrieval alone? (@10)

| Ordering | NDCG@10 | Recall@10 |
| --- | --- | --- |
| retrieval only | 0.0897 | 0.1070 |
| **+ ranker** | **0.0382 (-57.4%)** | 0.0498 |

Same 300 candidates, same targets, 300 users — only the ordering differs. The ranker changed the top result for 98% of them.

Recall@k is unchanged by construction when `candidate_k` is the size of the pool being reordered and k equals it; a difference here means the ranker moved relevant items into the top k.
# Ranker: does the negative sampler explain the missing resolution?

- Dataset: `synthetic`, variant `config`, vocabulary 945
- One shared retrieval model, its top 300 used both as the negative pool and as the slate every number below is measured on
- Baseline: `uniform` — uniform negatives, the configuration every earlier number in this repository was produced with

## The question

The heads explain most of the outcome variance on the positions they are trained on and almost none on the slates they are deployed on. If the cause is that uniform negatives make the training task too easy, then drawing negatives from retrieval's own candidates should move **resolution on the slate** — and nothing else in this table is the point.

## Resolution on the serving distribution

| Variant | click | cart | purchase | click (history) | purchase (history) |
| --- | --- | --- | --- | --- | --- |
| _retrieval, on its own slate_ | **0.17%** | **0.17%** | **0.09%** | – | – |
| `uniform` _(baseline)_ | 0.03% | 0.02% | 0.01% | 81.42% | 73.26% |
| `half_hard` | 0.01% | 0.01% | 0.01% | 81.57% | 73.10% |
| `all_hard` | 0.02% | 0.03% | 0.01% | 81.50% | 73.34% |

Resolution is the share of the outcome's variance a head's predictions explain. Calibration cannot change it, so it is the one number here that measures what the model knows rather than how it expresses it. The two right-hand columns are the same heads on history positions, unchanged by this experiment and shown so the gap stays visible.

**The first row is the control, and it is the one that makes the rest readable.** It is retrieval's own score, on the slate retrieval itself produced, against exactly the same per-head labels — the labelling code is shared between the two measurements precisely so this comparison cannot drift. Every number in this table is small in absolute terms, because most of the variance in *which one of 300 items a customer picks next* is irreducible. What matters is the ratio: if retrieval explains several times more of it than the model whose job is to reorder retrieval's output, the second stage is not adding knowledge, it is adding latency.

## Ordering, against retrieval and against the baseline

| Variant | blended MRR | vs retrieval | vs `uniform` (paired) | p | p adj | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| _retrieval_ | 0.6828 | 1.00x | – | – | – | – |
| `uniform` _(baseline)_ | 0.1389 | 0.20x | – | – | – | BLOCKED |
| `half_hard` | 0.0960 | 0.14x | 30.8% worse | 0.001 | 0.001 | BLOCKED |
| `all_hard` | 0.1312 | 0.19x | no difference detected | 0.611 | 0.611 | BLOCKED |

The comparison against the baseline is paired per user and sign-flipped, and the whole column is Holm-adjusted as one family. Two rankers scoring the same users differ far less between themselves than the users differ from each other, so an unpaired reading of these MRRs would call every row a tie.

## Re-ranking, the number that would actually ship

| Variant | retrieval NDCG@10 | re-ranked NDCG@10 | Δ |
| --- | --- | --- | --- |
| `uniform` | 0.0828 | 0.0169 | -79.5% |
| `half_hard` | 0.0828 | 0.0132 | -84.1% |
| `all_hard` | 0.0828 | 0.0203 | -75.5% |

One caveat this dataset cannot argue away: the vocabulary is 945 tokens and the slate is 300 of them, so a uniform negative already has roughly a one-in-three chance of being a candidate retrieval would have shown. Hard negatives have much less room to help here than they would on a real catalogue of millions, where a uniform draw is never a plausible item. A null result on this data is therefore weaker evidence against the hypothesis than a positive result would be for it.
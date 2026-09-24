# HSTU ablation on variant `config`

- Base config: `hstu_small.yaml`
- Token vocabulary: 945
- Test users: 943
- Metric: @200 on the test split
- Training seeds: 13, 17, 23

| Model | NDCG@200 | 95% interval | Final loss |
| --- | --- | --- | --- |
| `sasrec (baseline)` | 0.1790 | [0.1665, 0.1914] | 3.833 |
| `hstu_full` | 0.1756 | [0.1635, 0.1877] | 3.700 |
| `hstu_no_temporal_bias` | 0.1830 | [0.1709, 0.1951] | 3.778 |
| `hstu_no_rab` | 0.1850 | [0.1729, 0.1972] | 3.783 |
| `hstu_no_actions` | 0.1690 | [0.1567, 0.1815] | 3.720 |
| `hstu_seq_normalised` | 0.1776 | [0.1653, 0.1896] | 3.716 |
| `hstu_all_targets` | 0.1771 | [0.1651, 0.1889] | 3.707 |

## Against the SASRec baseline — NDCG@10, paired on the same users

| Model | Δ NDCG@10 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `hstu_full` | -0.0022 | [-0.0055, +0.0010] | 0.196 | 1.000 | no difference detected |
| `hstu_no_temporal_bias` | -0.0013 | [-0.0044, +0.0018] | 0.399 | 1.000 | no difference detected |
| `hstu_no_rab` | -0.0003 | [-0.0036, +0.0029] | 0.870 | 1.000 | no difference detected |
| `hstu_no_actions` | -0.0006 | [-0.0036, +0.0022] | 0.661 | 1.000 | no difference detected |
| `hstu_seq_normalised` | -0.0016 | [-0.0049, +0.0015] | 0.322 | 1.000 | no difference detected |
| `hstu_all_targets` | -0.0017 | [-0.0051, +0.0017] | 0.360 | 1.000 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the full HSTU — NDCG@10, paired on the same users

| Model | Δ NDCG@10 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | +0.0022 | [-0.0010, +0.0055] | 0.196 | 0.980 | no difference detected |
| `hstu_no_temporal_bias` | +0.0009 | [-0.0018, +0.0036] | 0.556 | 1.000 | no difference detected |
| `hstu_no_rab` | +0.0019 | [-0.0009, +0.0050] | 0.232 | 0.980 | no difference detected |
| `hstu_no_actions` | +0.0016 | [-0.0005, +0.0038] | 0.149 | 0.894 | no difference detected |
| `hstu_seq_normalised` | +0.0006 | [-0.0015, +0.0028] | 0.601 | 1.000 | no difference detected |
| `hstu_all_targets` | +0.0006 | [-0.0004, +0.0015] | 0.269 | 0.980 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the SASRec baseline — NDCG@200, paired on the same users

| Model | Δ NDCG@200 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `hstu_full` | -0.0033 | [-0.0081, +0.0010] | 0.164 | 0.493 | no difference detected |
| `hstu_no_temporal_bias` | +0.0040 | [-0.0003, +0.0083] | 0.082 | 0.328 | no difference detected |
| `hstu_no_rab` | +0.0060 | [+0.0018, +0.0102] | 0.009 | 0.045 | **3.3% better at @200** |
| `hstu_no_actions` | -0.0100 | [-0.0147, -0.0054] | 0.001 | 0.003 | **5.6% worse at @200** |
| `hstu_seq_normalised` | -0.0014 | [-0.0061, +0.0032] | 0.608 | 0.907 | no difference detected |
| `hstu_all_targets` | -0.0019 | [-0.0067, +0.0026] | 0.453 | 0.907 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the full HSTU — NDCG@200, paired on the same users

| Model | Δ NDCG@200 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | +0.0033 | [-0.0010, +0.0081] | 0.164 | 0.270 | no difference detected |
| `hstu_no_temporal_bias` | +0.0074 | [+0.0041, +0.0105] | 0.001 | 0.003 | **4.2% better at @200** |
| `hstu_no_rab` | +0.0093 | [+0.0052, +0.0134] | 0.001 | 0.003 | **5.3% better at @200** |
| `hstu_no_actions` | -0.0066 | [-0.0096, -0.0036] | 0.001 | 0.003 | **3.8% worse at @200** |
| `hstu_seq_normalised` | +0.0020 | [-0.0005, +0.0046] | 0.135 | 0.270 | no difference detected |
| `hstu_all_targets` | +0.0014 | [+0.0003, +0.0027] | 0.027 | 0.081 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Across training seeds

| Model | Seeds | NDCG@200 mean | min | max | std |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | 3 | 0.1802 | 0.1790 | 0.1809 | 0.0011 |
| `hstu_full` | 3 | 0.1757 | 0.1731 | 0.1783 | 0.0026 |
| `hstu_no_temporal_bias` | 3 | 0.1833 | 0.1818 | 0.1850 | 0.0016 |
| `hstu_no_rab` | 3 | 0.1828 | 0.1817 | 0.1850 | 0.0018 |
| `hstu_no_actions` | 3 | 0.1704 | 0.1690 | 0.1720 | 0.0015 |
| `hstu_seq_normalised` | 3 | 0.1773 | 0.1755 | 0.1789 | 0.0017 |
| `hstu_all_targets` | 3 | 0.1765 | 0.1735 | 0.1790 | 0.0028 |

This is the variance a bootstrap cannot see. If the spread between seeds of the *same* configuration is as large as the difference between two configurations, the difference is not a finding.

Each row changes exactly one thing against the base config. A row that beats `hstu_full` *with an interval that excludes zero* means that component is not earning its place on this data — **at that cutoff**. A finding at NDCG@200 is a statement about positions 11-200, a fifth of a 945-item catalogue; it says nothing about the ten a customer sees unless the @10 table agrees.
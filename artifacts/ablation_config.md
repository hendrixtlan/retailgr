# HSTU ablation on variant `config`

- Base config: `hstu_small.yaml`
- Token vocabulary: 945
- Test users: 943
- Metric: @200 on the test split
- Training seeds: 13, 17, 23

| Model | NDCG@200 | 95% interval | Final loss |
| --- | --- | --- | --- |
| `sasrec (baseline)` | 0.1889 | [0.1777, 0.2004] | 3.833 |
| `hstu_full` | 0.1816 | [0.1692, 0.1939] | 3.533 |
| `hstu_no_temporal_bias` | 0.1883 | [0.1777, 0.1994] | 3.778 |
| `hstu_no_rab` | 0.1883 | [0.1765, 0.2003] | 3.734 |
| `hstu_no_actions` | 0.1702 | [0.1581, 0.1825] | 3.508 |
| `hstu_seq_normalised` | 0.1854 | [0.1733, 0.1975] | 3.653 |
| `hstu_all_targets` | 0.1822 | [0.1699, 0.1946] | 3.541 |

## Against the SASRec baseline — NDCG@10, paired on the same users

| Model | Δ NDCG@10 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `hstu_full` | +0.0026 | [-0.0017, +0.0069] | 0.232 | 1.000 | no difference detected |
| `hstu_no_temporal_bias` | +0.0009 | [-0.0040, +0.0059] | 0.747 | 1.000 | no difference detected |
| `hstu_no_rab` | +0.0039 | [-0.0005, +0.0082] | 0.077 | 0.462 | no difference detected |
| `hstu_no_actions` | +0.0014 | [-0.0033, +0.0061] | 0.552 | 1.000 | no difference detected |
| `hstu_seq_normalised` | +0.0019 | [-0.0024, +0.0064] | 0.396 | 1.000 | no difference detected |
| `hstu_all_targets` | +0.0024 | [-0.0020, +0.0068] | 0.281 | 1.000 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the full HSTU — NDCG@10, paired on the same users

| Model | Δ NDCG@10 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | -0.0026 | [-0.0069, +0.0017] | 0.232 | 1.000 | no difference detected |
| `hstu_no_temporal_bias` | -0.0017 | [-0.0058, +0.0023] | 0.433 | 1.000 | no difference detected |
| `hstu_no_rab` | +0.0013 | [-0.0012, +0.0039] | 0.331 | 1.000 | no difference detected |
| `hstu_no_actions` | -0.0012 | [-0.0034, +0.0009] | 0.302 | 1.000 | no difference detected |
| `hstu_seq_normalised` | -0.0007 | [-0.0029, +0.0016] | 0.558 | 1.000 | no difference detected |
| `hstu_all_targets` | -0.0002 | [-0.0013, +0.0009] | 0.750 | 1.000 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the SASRec baseline — NDCG@200, paired on the same users

| Model | Δ NDCG@200 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `hstu_full` | -0.0073 | [-0.0136, -0.0018] | 0.015 | 0.075 | no difference detected |
| `hstu_no_temporal_bias` | -0.0005 | [-0.0065, +0.0051] | 0.856 | 1.000 | no difference detected |
| `hstu_no_rab` | -0.0006 | [-0.0056, +0.0047] | 0.817 | 1.000 | no difference detected |
| `hstu_no_actions` | -0.0186 | [-0.0251, -0.0126] | 0.001 | 0.003 | **9.9% worse at @200** |
| `hstu_seq_normalised` | -0.0034 | [-0.0090, +0.0018] | 0.213 | 0.639 | no difference detected |
| `hstu_all_targets` | -0.0066 | [-0.0128, -0.0011] | 0.027 | 0.108 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Against the full HSTU — NDCG@200, paired on the same users

| Model | Δ NDCG@200 | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | +0.0073 | [+0.0018, +0.0136] | 0.015 | 0.030 | **4.0% better at @200** |
| `hstu_no_temporal_bias` | +0.0068 | [+0.0024, +0.0114] | 0.009 | 0.029 | **3.7% better at @200** |
| `hstu_no_rab` | +0.0067 | [+0.0025, +0.0109] | 0.002 | 0.007 | **3.7% better at @200** |
| `hstu_no_actions` | -0.0113 | [-0.0146, -0.0082] | 0.001 | 0.003 | **6.2% worse at @200** |
| `hstu_seq_normalised` | +0.0039 | [+0.0012, +0.0068] | 0.006 | 0.024 | **2.1% better at @200** |
| `hstu_all_targets` | +0.0007 | [-0.0004, +0.0018] | 0.212 | 0.212 | no difference detected |

`p adj.` is Holm-Bonferroni across the 6 comparisons in this table; the verdict uses it. A table of tests read at p<0.05 each is several chances to be fooled, not one.

## Across training seeds

| Model | Seeds | NDCG@200 mean | min | max | std |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | 3 | 0.1853 | 0.1786 | 0.1889 | 0.0059 |
| `hstu_full` | 3 | 0.1820 | 0.1731 | 0.1912 | 0.0090 |
| `hstu_no_temporal_bias` | 3 | 0.1872 | 0.1836 | 0.1897 | 0.0032 |
| `hstu_no_rab` | 3 | 0.1851 | 0.1807 | 0.1883 | 0.0039 |
| `hstu_no_actions` | 3 | 0.1709 | 0.1698 | 0.1726 | 0.0015 |
| `hstu_seq_normalised` | 3 | 0.1829 | 0.1745 | 0.1889 | 0.0075 |
| `hstu_all_targets` | 3 | 0.1824 | 0.1735 | 0.1916 | 0.0091 |

This is the variance a bootstrap cannot see. If the spread between seeds of the *same* configuration is as large as the difference between two configurations, the difference is not a finding.

Each row changes exactly one thing against the base config. A row that beats `hstu_full` *with an interval that excludes zero* means that component is not earning its place on this data — **at that cutoff**. A finding at NDCG@200 is a statement about positions 11-200, a fifth of a 945-item catalogue; it says nothing about the ten a customer sees unless the @10 table agrees.
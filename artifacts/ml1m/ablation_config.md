# HSTU ablation on variant `config`

- Base config: `hstu_ml1m.yaml`
- Token vocabulary: 3663
- Metric: @200 on the test split

| Model | Recall@200 | NDCG@200 | vs SASRec | Fit (s) | Final loss |
| --- | --- | --- | --- | --- | --- |
| `sasrec (baseline)` | 0.3090 | 0.2087 | baseline | 590.0 | 2.850 |
| `hstu_full` | 0.2990 | 0.2049 | -1.8% | 493.0 | 2.688 |
| `hstu_no_temporal_bias` | 0.3106 | 0.2143 | +2.7% | 429.5 | 2.837 |

Each row changes exactly one thing against the base config, seed included. A row that beats `hstu_full` means that component is not earning its place on this data.
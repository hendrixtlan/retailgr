# RetailGR - Stage 1 run report

- Dataset: `synthetic`
- Warehouse backend: `parquet`
- Started (UTC): 2026-09-20T14:25:11+00:00
- Duration: 339.27s

## Vocabulary and sparsity

| Variant | Tokens | Distinct SKUs | SKUs per token | Tokens < 5 events |
| --- | --- | --- | --- | --- |
| `sku` | 3033 | 3081 | 1.016 | 858 (28.3%) |
| `style_color` | 809 | 3081 | 3.808 | 16 (2.0%) |
| `product` | 499 | 3081 | 6.174 | 8 (1.6%) |
| `config` | 944 | 3081 | 3.264 | 36 (3.8%) |

## Test metrics @200

Means with 95% bootstrap intervals over users. The interval covers user sampling only — not training-seed variance, which needs several runs (`ablate --seeds`).

| Variant | Model | Recall@200 | NDCG@200 | Coverage@200 | Users |
| --- | --- | --- | --- | --- | --- |
| `sku` | popularity | 0.3261 [0.3109, 0.3412] | 0.1259 [0.1185, 0.1337] | 0.0748 | 1100 |
| `sku` | sasrec_small | 0.2926 [0.2744, 0.3108] | 0.1304 [0.1205, 0.1409] | 0.8042 | 1100 |
| `sku` | hstu_small | 0.2828 [0.2643, 0.3011] | 0.1268 [0.1169, 0.1371] | 0.8583 | 1100 |
| `style_color` | popularity | 0.4201 [0.4024, 0.4372] | 0.1541 [0.1459, 0.1620] | 0.2802 | 1101 |
| `style_color` | sasrec_small | 0.4218 [0.4023, 0.4401] | 0.1813 [0.1701, 0.1924] | 0.9988 | 1101 |
| `style_color` | hstu_small | 0.4161 [0.3975, 0.4350] | 0.1799 [0.1682, 0.1910] | 0.9988 | 1101 |
| `product` | popularity | 0.5326 [0.5128, 0.5524] | 0.1875 [0.1787, 0.1962] | 0.4640 | 1101 |
| `product` | sasrec_small | 0.5294 [0.5109, 0.5479] | 0.2227 [0.2115, 0.2338] | 0.9980 | 1101 |
| `product` | hstu_small | 0.4736 [0.4542, 0.4924] | 0.2080 [0.1963, 0.2196] | 0.9980 | 1101 |
| `config` | popularity | 0.4095 [0.3923, 0.4276] | 0.1509 [0.1429, 0.1589] | 0.2402 | 1101 |
| `config` | sasrec_small | 0.4219 [0.4031, 0.4414] | 0.1797 [0.1682, 0.1909] | 0.9979 | 1101 |
| `config` | hstu_small | 0.4135 [0.3941, 0.4323] | 0.1758 [0.1647, 0.1868] | 0.9989 | 1101 |

## Head to head vs sasrec_small, NDCG@200

| Variant | Challenger | Difference | 95% interval | p | p adj. | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `sku` | hstu_small | -0.0037 | [-0.0066, -0.0007] | 0.020 | 0.061 | no difference detected |
| `style_color` | hstu_small | -0.0014 | [-0.0059, +0.0033] | 0.562 | 0.562 | no difference detected |
| `product` | hstu_small | -0.0147 | [-0.0197, -0.0098] | 0.001 | 0.002 | **6.6% worse** |
| `config` | hstu_small | -0.0039 | [-0.0081, +0.0006] | 0.087 | 0.174 | no difference detected |

Paired against `sasrec_small` on the same users, so the between-user spread — which is much larger than the gap between two models — is removed. That pairing is what gives these comparisons the power to resolve a 3% difference at all.

`p` is a sign-flip permutation test; `p adj.` is Holm-Bonferroni across the 4 comparisons in this table, because showing several tests and reading the small ones is several chances to be fooled rather than one. The verdict column uses the adjusted value.

The interval is on the *difference*. When it includes zero the two models are indistinguishable on this data, whatever the point estimates look like. None of this covers training-seed variance — for that, `ablate --seeds`.

The presets are matched on hidden size, depth, heads, dropout, sequence length, loss, batch size, epochs and seed, so a real gap is the architecture and its extra modalities, not capacity.

## sasrec_small by category, Recall@200

| Variant | apparel | electronics | footwear | grocery | home |
| --- | --- | --- | --- | --- | --- |
| `sku` | 0.3111 | 0.2956 | 0.2735 | 0.4096 | 0.3164 |
| `style_color` | 0.4296 | 0.5057 | 0.4094 | 0.4850 | 0.4450 |
| `product` | 0.5265 | 0.5470 | 0.5273 | 0.5374 | 0.5625 |
| `config` | 0.4233 | 0.4200 | 0.4206 | 0.4778 | 0.4193 |

## hstu_small by category, Recall@200

| Variant | apparel | electronics | footwear | grocery | home |
| --- | --- | --- | --- | --- | --- |
| `sku` | 0.2930 | 0.3673 | 0.2673 | 0.3843 | 0.3127 |
| `style_color` | 0.4278 | 0.4569 | 0.3993 | 0.4252 | 0.4664 |
| `product` | 0.4725 | 0.6039 | 0.4705 | 0.6047 | 0.4597 |
| `config` | 0.4228 | 0.4775 | 0.4004 | 0.4530 | 0.4410 |

For the granularity decision, read the per-category tables rather than the overall one: the winning token level is a per-category choice, and the overall number is dominated by whichever category has the most traffic.
# RetailGR - Stage 1 run report

- Dataset: `ml1m`
- Warehouse backend: `parquet`
- Started (UTC): 2026-09-18T16:16:57+00:00
- Duration: 1850.66s

## Vocabulary and sparsity

| Variant | Tokens | Distinct SKUs | SKUs per token | Tokens < 5 events |
| --- | --- | --- | --- | --- |
| `config` | 3662 | 3706 | 1.012 | 315 (8.6%) |

## Test metrics @200

| Variant | Model | Recall@200 | NDCG@200 | HitRate@200 | Coverage@200 | Users |
| --- | --- | --- | --- | --- | --- | --- |
| `config` | popularity | 0.3273 | 0.2310 | 0.9277 | 0.0893 | 1175 |
| `config` | sasrec_ml1m | 0.3062 | 0.2063 | 0.9081 | 0.6803 | 1175 |
| `config` | hstu_ml1m | 0.2991 | 0.2038 | 0.8868 | 0.6366 | 1175 |

## Head to head vs sasrec_ml1m, NDCG@200

| Variant | sasrec_ml1m | hstu_ml1m | Best |
| --- | --- | --- | --- |
| `config` | 0.2063 | 0.2038 (-1.2%) | sasrec_ml1m |

Percentages are relative to `sasrec_ml1m` on the same variant, split and metric. The presets are matched on hidden size, depth, heads, dropout, sequence length, loss, batch size, epochs and seed, so a gap is the architecture and its extra modalities, not capacity.

## sasrec_ml1m by category, Recall@200

| Variant | movies |
| --- | --- |
| `config` | 0.3062 |

## hstu_ml1m by category, Recall@200

| Variant | movies |
| --- | --- |
| `config` | 0.2991 |

For the granularity decision, read the per-category tables rather than the overall one: the winning token level is a per-category choice, and the overall number is dominated by whichever category has the most traffic.
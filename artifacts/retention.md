# Retention

- Mode: dry run
- Clock: **data**
- Refuses above: 50% of a table
- Duration: 22.3s

| Layer | Window | Status | Removed | Share |
| --- | ---: | --- | ---: | ---: |
| `bronze.interactions` | 45d | would prune | 41880 | 46.3% |
| `silver.interactions` | 400d | would prune | 0 | 0.0% |
| `gold.sequences_config` | 400d | would prune | 0 | 0.0% |
| `gold.sequences_product` | 400d | absent | - | - |
| `gold.sequences_sku` | 400d | absent | - | - |
| `gold.sequences_style_color` | 400d | absent | - | - |

## Snapshot expiry

- not applicable: parquet rewrites in place

Without this the pruning above is cosmetic: in Iceberg a delete writes a new snapshot and every earlier one stays readable by id.

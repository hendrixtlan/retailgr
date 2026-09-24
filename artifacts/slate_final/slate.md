# Ordering the serving slate

- Variant: `config`, candidate_k: 300
- Seeds per arm: [13, 17, 23], users: 400

AUC, not resolution share. Resolution is quadratic in the deviation from the base rate, and the slate's base rate is ~0.2% against the teacher-forced set's 24% — the same ordering scores ~144x lower on the slate from that alone.

## click

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5437 | ±0.0015 | +0.0437 | 1.00x | no | 1.42x |
| `no_history` | 0.6367 | ±0.0111 | +0.1367 | 3.13x | yes | 0.65x |
| `return_only` | 0.5130 | ±0.0093 | +0.0130 | 0.30x | yes | 0.34x |

## cart

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5399 | ±0.0108 | +0.0399 | 1.00x | no | 1.51x |
| `no_history` | 0.6377 | ±0.0091 | +0.1377 | 3.45x | yes | 0.78x |
| `return_only` | 0.5059 | ±0.0103 | +0.0059 | 0.15x | yes | 0.37x |

## purchase

Retrieval's own ordering of the same candidates: **AUC 0.6953**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5025 | ±0.0338 | +0.0025 | 1.00x | no | 1.94x |
| `no_history` | 0.5774 | ±0.0081 | +0.0774 | 30.96x | yes | 1.28x |
| `return_only` | 0.4920 | ±0.0212 | -0.0080 | -3.20x | no | 0.51x |

## return

No slate-shaped label exists for this head — a slate yields at most one
matured purchase per user — so it is measured on history positions, the
only distribution where it exists. An arm that wins above and loses here
is trading a head away where the main table cannot see it.

| Arm | Return head AUC |
| --- | ---: |
| `control` | 0.8500 |
| `no_history` | 0.5001 |
| `return_only` | 0.8455 |


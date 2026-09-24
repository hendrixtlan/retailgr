# Ordering the serving slate

- Variant: `config`, candidate_k: 300
- Seeds per arm: [13, 17, 23], users: 400

AUC, not resolution share. Resolution is quadratic in the deviation from the base rate, and the slate's base rate is ~0.2% against the teacher-forced set's 24% — the same ordering scores ~144x lower on the slate from that alone.

## click

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5437 | ±0.0015 | +0.0437 | 1.00x | no | 1.42x |
| `scale_50` | 0.5690 | ±0.0156 | +0.0690 | 1.58x | yes | 2.49x |
| `scale_25` | 0.5775 | ±0.0051 | +0.0775 | 1.77x | yes | 2.65x |
| `scale_10` | 0.6287 | ±0.0107 | +0.1287 | 2.94x | yes | 2.71x |
| `no_history` | 0.6367 | ±0.0111 | +0.1367 | 3.13x | yes | 0.65x |

## cart

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5399 | ±0.0108 | +0.0399 | 1.00x | no | 1.51x |
| `scale_50` | 0.5696 | ±0.0155 | +0.0696 | 1.74x | yes | 2.52x |
| `scale_25` | 0.5803 | ±0.0074 | +0.0803 | 2.01x | yes | 2.60x |
| `scale_10` | 0.6272 | ±0.0165 | +0.1272 | 3.19x | yes | 2.87x |
| `no_history` | 0.6377 | ±0.0091 | +0.1377 | 3.45x | yes | 0.78x |

## purchase

Retrieval's own ordering of the same candidates: **AUC 0.6953**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5025 | ±0.0338 | +0.0025 | 1.00x | no | 1.94x |
| `scale_50` | 0.5253 | ±0.0368 | +0.0253 | 10.12x | no | 2.61x |
| `scale_25` | 0.5243 | ±0.0296 | +0.0243 | 9.72x | no | 3.49x |
| `scale_10` | 0.5760 | ±0.0169 | +0.0760 | 30.40x | yes | 2.59x |
| `no_history` | 0.5774 | ±0.0081 | +0.0774 | 30.96x | yes | 1.28x |

## return

No slate-shaped label exists for this head — a slate yields at most one
matured purchase per user — so it is measured on history positions, the
only distribution where it exists. An arm that wins above and loses here
is trading a head away where the main table cannot see it.

| Arm | Return head AUC |
| --- | ---: |
| `control` | 0.8500 |
| `scale_50` | 0.8273 |
| `scale_25` | 0.8279 |
| `scale_10` | 0.7195 |
| `no_history` | 0.5001 |


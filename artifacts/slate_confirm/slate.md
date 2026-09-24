# Ordering the serving slate

- Variant: `config`, candidate_k: 300
- Seeds per arm: [13, 17, 23, 29, 31], users: 400

AUC, not resolution share. Resolution is quadratic in the deviation from the base rate, and the slate's base rate is ~0.2% against the teacher-forced set's 24% — the same ordering scores ~144x lower on the slate from that alone.

## click

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5455 | ±0.0134 | +0.0455 | 1.00x | no | 1.97x |
| `no_history` | 0.6312 | ±0.0110 | +0.1312 | 2.88x | yes | 0.51x |

## cart

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5463 | ±0.0151 | +0.0463 | 1.00x | no | 2.03x |
| `no_history` | 0.6315 | ±0.0109 | +0.1315 | 2.84x | yes | 0.59x |

## purchase

Retrieval's own ordering of the same candidates: **AUC 0.6953**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `control` | 0.5196 | ±0.0375 | +0.0196 | 1.00x | no | 2.35x |
| `no_history` | 0.5786 | ±0.0065 | +0.0786 | 4.01x | no | 0.92x |


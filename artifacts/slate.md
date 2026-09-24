# Ordering the serving slate

- Variant: `config`, candidate_k: 300
- Seeds per arm: [13, 17, 23], users: 400

AUC, not resolution share. Resolution is quadratic in the deviation from the base rate, and the slate's base rate is ~0.2% against the teacher-forced set's 24% — the same ordering scores ~144x lower on the slate from that alone.

## click

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise |
| --- | ---: | ---: | ---: | ---: | --- |
| `control` | 0.5437 | ±0.0015 | +0.0437 | 1.00x | no |
| `unseen` | 0.5245 | ±0.0461 | +0.0245 | 0.56x | yes |
| `no_history` | 0.6367 | ±0.0111 | +0.1367 | 3.13x | yes |
| `both` | 0.5366 | ±0.0354 | +0.0366 | 0.84x | yes |

## cart

Retrieval's own ordering of the same candidates: **AUC 0.7138**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise |
| --- | ---: | ---: | ---: | ---: | --- |
| `control` | 0.5399 | ±0.0108 | +0.0399 | 1.00x | no |
| `unseen` | 0.5258 | ±0.0499 | +0.0258 | 0.65x | no |
| `no_history` | 0.6377 | ±0.0091 | +0.1377 | 3.45x | yes |
| `both` | 0.5350 | ±0.0095 | +0.0350 | 0.88x | no |

## purchase

Retrieval's own ordering of the same candidates: **AUC 0.6953**. A second stage below this line has no reason to exist.

| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise |
| --- | ---: | ---: | ---: | ---: | --- |
| `control` | 0.5025 | ±0.0338 | +0.0025 | 1.00x | no |
| `unseen` | 0.4959 | ±0.0199 | -0.0041 | -1.64x | no |
| `no_history` | 0.5774 | ±0.0081 | +0.0774 | 30.96x | yes |
| `both` | 0.5135 | ±0.0253 | +0.0135 | 5.40x | no |

`unseen` fell back to the last event for **0.4%** of users, who have no position whose item is new. That is supervision the arm gives up, and it is the cost of matching the serving distribution.


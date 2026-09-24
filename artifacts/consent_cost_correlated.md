# What consent costs

- Variant: `config`, model: `hstu_small.yaml`
- Seeds per point: [0, 1, 2]
- Training users at 100%: 2501
- Opt-out selection: **most active first**

| Opt-in rate | Train users | Recall@10 | Seed spread | vs. 100% | Beyond noise |
| --- | ---: | ---: | ---: | ---: | --- |
| 100% | 2501 | 0.0955 | ±0.0013 | 1.000x | no |
| 90% | 2251 | 0.0957 | ±0.0014 | 1.002x | no |
| 75% | 1876 | 0.0954 | ±0.0009 | 0.999x | no |
| 50% | 1250 | 0.0905 | ±0.0015 | 0.947x | yes |
| 25% | 625 | 0.0758 | ±0.0035 | 0.794x | yes |

**Read the last column first.** This model's seed-to-seed spread is not
small, and a difference inside it is not a difference. A row marked `no`
means the measured drop at that opt-in rate is indistinguishable from
re-running the same configuration with a different random seed.

**The pessimistic direction.** Users are dropped most-active-first,
which is a guess about who opts out, not a finding. Its value is the
comparison with the uniform run: the gap between the two curves is the
share of the cost that comes from *who* leaves rather than *how many*.

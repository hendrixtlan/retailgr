# What consent costs

- Variant: `config`, model: `hstu_small.yaml`
- Seeds per point: [0, 1, 2]
- Training users at 100%: 2501
- Opt-out selection: **uniform**

| Opt-in rate | Train users | Recall@10 | Seed spread | vs. 100% | Beyond noise |
| --- | ---: | ---: | ---: | ---: | --- |
| 100% | 2501 | 0.0955 | ±0.0013 | 1.000x | no |
| 90% | 2251 | 0.0952 | ±0.0005 | 0.997x | no |
| 75% | 1876 | 0.0945 | ±0.0013 | 0.989x | no |
| 50% | 1250 | 0.0927 | ±0.0012 | 0.971x | yes |
| 25% | 625 | 0.0810 | ±0.0060 | 0.849x | yes |

**Read the last column first.** This model's seed-to-seed spread is not
small, and a difference inside it is not a difference. A row marked `no`
means the measured drop at that opt-in rate is indistinguishable from
re-running the same configuration with a different random seed.

**This is a lower bound.** Users are dropped uniformly, which assumes
consent is independent of behaviour. It is not: privacy-conscious
customers differ from the average in ways that are hard to
characterise and easy to get backwards. Run with `--correlate` to see
how much of the answer depends on *who* opts out rather than how many.

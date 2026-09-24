# Where the models converge

- Variant: `config`, seeds: [13, 17, 23]
- Early stopping on validation `ndcg@10`, patience 10, cap 150 epochs
- Both schedules scored on **test**; validation only chose the epoch.

| Model | Best epoch per seed | 8 epochs | Converged | Gain |
| --- | --- | ---: | ---: | ---: |
| `sasrec_small` | 3, 3, 7 | 0.0857 ±0.0003 | 0.0845 ±0.0012 | -1.4% |
| `hstu_small` | 6, 8, 3 | 0.0843 ±0.0004 | 0.0841 ±0.0020 | -0.2% |

## Converged against eight epochs, paired per user

- `sasrec_small`: -0.0012 [-0.0040, +0.0015], p=0.401
- `hstu_small`: -0.0001 [-0.0016, +0.0013], p=0.845

## The model comparison, at both schedules

- fixed: `sasrec_small` − `hstu_small` = +0.0015 [-0.0005, +0.0034], p=0.172
- stopped: `sasrec_small` − `hstu_small` = +0.0004 [-0.0024, +0.0033], p=0.779

The validation curve is not quoted as a result: once it chooses the epoch, its best point is the maximum of many noisy looks and is biased upwards. Test is touched once per run, after the choice.

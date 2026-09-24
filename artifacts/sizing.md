# Deployment footprint

Measured, on the bundle that would actually be served. Every `resources:` block in `deploy/k8s/` carries these numbers rather than a figure copied from another manifest.

## The bundle

**1 MiB** total.

| File | Size |
| --- | --- |
| `hierarchy.json` | 98 KiB |
| `item_embeddings.npy` | 236 KiB |
| `manifest.json` | 107 KiB |
| `model.pt` | 407 KiB |
| `ranker.pt` | 478 KiB |
| `vocab.json` | 39 KiB |

## Memory

| Measurement | Value |
| --- | --- |
| A bare interpreter | 11 MiB |
| After importing the serving stack | 29 MiB _(+17 MiB)_ |
| After loading the bundle (torch loads here, lazily) | 510 MiB _(+481 MiB)_ |
| Warm, after 25 requests | **523 MiB** _(+13 MiB)_ |

Worth reading the third row before anyone tries to shrink the model to fit a pod: the bundle on disk is 1.3 MiB and loading it costs 481 MiB, because that step is where PyTorch actually initialises. Almost all of this pod is framework, and none of that part gets smaller by training a smaller model.

## Cold start

| Phase | Seconds |
| --- | --- |
| Import | 0.08 |
| Load the bundle | 1.11 |
| First request | 0.02 |
| **Total** | **1.23** |

Median of 3 fresh interpreters. This is what a readiness probe has to clear and what an autoscaler pays for every replica it adds.

## Throughput

417 req/s on 2 cores — **208 req/s per core**, without the ranker in the path.

In-process, so it excludes HTTP framing and the network. It is the ceiling the request path imposes, not a number to promise anyone.

## What the manifests should say

```yaml
resources:
  requests:
    memory: 655Mi   # measured warm 523.2 MiB + 25%
  limits:
    memory: 1047Mi   # 2x warm: a limit is a kill switch, not a target
readinessProbe:
  initialDelaySeconds: 5   # measured cold start 1.226s, doubled
```

No CPU limit is suggested. A CPU limit throttles rather than kills, and throttling a latency-budgeted request path trades a clean autoscaling signal for p99 spikes that look like a model problem.
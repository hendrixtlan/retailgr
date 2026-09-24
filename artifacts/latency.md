# Serving latency

- Model: `hstu` (`hstu_small-config-945`)
- Dataset: `synthetic`, variant `config`
- Retrieval index: 945 items, top-800 then 300 to the ranker
- 500 requests over 525 real user histories, after 25 warm-up calls
- Throughput: 279.1 req/s, single process, single thread

In-process measurement: it covers context assembly, retrieval, the model
forward pass and the policy layer, but not HTTP or network.

## Per stage

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | Budget (ms) | Verdict |
| --- | --- | --- | --- | --- | --- |
| `context` | 0.044 | 0.075 | 0.089 | 10 | within |
| `retrieval` | 1.733 | 1.907 | 2.217 | 25 | within |
| `filter` | 0.243 | 0.303 | 0.349 | 5 | within |
| `ranking` | 0.002 | 0.002 | 0.011 | 35 | within |
| `rerank` | 1.010 | 1.166 | 1.309 | 5 | within |
| **end to end** | **3.454** | **3.790** | **4.572** | **100** | **within** |

Responses by path: model: 500.

Mean items returned: 20.0 of 20 requested.
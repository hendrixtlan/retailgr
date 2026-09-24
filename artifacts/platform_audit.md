# Platform audit

The architecture claim — *decoupled compute, storage and machine learning, so it runs on any cloud without proprietary platforms* — checked mechanically instead of asserted.

**PASSED**

| Check | Findings | Verdict |
| --- | --- | --- |
| No cloud vendor SDK in application code | 0 | pass |
| No endpoint configuration cannot reach | 0 | pass |
| Optional clients imported lazily | 0 | pass |
| Backends conform to their protocol | 0 | pass |

## Swap points

| Component | Setting | Backend | Needs | Conforms |
| --- | --- | --- | --- | --- |
| `warehouse` | `warehouse.backend` | `parquet` _(default)_ | core only | yes |
| `warehouse` | `warehouse.backend` | `iceberg` | core only | yes |
| `broker` | `streaming.broker` | `memory` _(default)_ | core only | yes |
| `broker` | `streaming.broker` | `kafka` | `kafka` | yes |
| `online_store` | `online_store.backend` | `memory` _(default)_ | core only | yes |
| `online_store` | `online_store.backend` | `redis` | `redis` | yes |
| `retrieval_index` | `serving.index` | `exact` _(default)_ | core only | yes |
| `retrieval_index` | `serving.index` | `faiss` | `faiss` | yes |

Conformance is checked statically — by comparing each backend's methods and signatures against the `Protocol` it claims to satisfy — so it runs with no Kafka, Redis or FAISS present. Python checks neither of those things at runtime, which means a backend can be missing a method for as long as nobody selects it. That is precisely the failure a portability claim is supposed to rule out, and the one an integration test cannot catch without the infrastructure it is testing.

What this does **not** establish: that the system runs on any particular cloud. It establishes the absence of the specific things that would stop it — a vendor SDK in the request path, an address nobody can change, a backend that is swappable only in the README.
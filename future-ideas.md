
## Embedding pool (was: peer-to-peer embeddings)

Embedding each song can take a while — for libraries with thousands of tracks it can take days. A centralized **embedding pool** lets users download community consensus vectors before running local GPU work, and optionally contribute vectors after embedding.

**Spec:** [docs/embedding-pool-spec.md](docs/embedding-pool-spec.md)

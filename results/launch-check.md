# Mechanical launch check

scripts used during tuning compared `DRY_RUN=1 ./serve.sh PROFILE` with the launches that were measured:
docker flags, environment, image digest, vLLM arguments, and every mounted file byte for byte (after its change notice).

```
fastest  vs configs/fin3c.json: IDENTICAL (6 env, 16 mounts, 24 vLLM args); /patches/dvocab dir mount -> file mount of draft_vocab.json: same bytes
fast     vs configs/gsf-f2.json: IDENTICAL (6 env, 13 mounts, 24 vLLM args); /patches/dvocab dir mount -> file mount of draft_vocab.json: same bytes
lossless vs configs/fin-g.json: IDENTICAL (4 env, 11 mounts, 24 vLLM args); /patches/dvocab dir mount -> file mount of draft_vocab.json: same bytes
```

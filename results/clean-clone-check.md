# Clean-clone check, 2026-11-24 (UTC)

git clone of this repository (commit 75add98), default profile, PORT=8001, empty CACHE_DIR.
```
10:17:15 A: cold start, CACHE_DIR=<empty dir>
starting profile 'fastest' as inkling on 127.0.0.1:8001
ready after 224 s: http://127.0.0.1:8001/v1 (model thinkingmachines/Inkling-2-Mini)
PASS  /v1/models: ['thinkingmachines/Inkling-2-Mini']
PASS  chat with reasoning: reasoning 679 chars, answer '17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391.\n\n391'
PASS  tool call: get_weather({'city': 'Paris'})
PASS  tool result used: 'Right now in Paris, it’s 18°C and cloudy.'
PASS  image input: 'red'
KL vs measured fin3c: positions 11214  KL mean 0.00000  p99 0.0000  max 0.000  NLL 1.0152 -> 1.0152 (delta +0.0000)  top-1 agreement 100.00%
10:21:45 c1 median 61.13 [61.13]
10:21:47 prefill ~4096 2081.1 tok/s over 4107 tokens
10:21:54 prefill ~32768 4294.6 tok/s over 31945 tokens
 "ttft_short_ms": 162.1,
 "mem_peak_gib": 96.2,
draining inkling (up to 120 s)
stopped
```

The short bench above ran right after the cold start, so its 4K prompt includes kernel compilation.

# Results

| file | what |
|---|---|
| [headline.md](headline.md) | the confirmation run of the three profiles and stock (the README's headline table) |
| [confirmation.json](confirmation.json) | the same numbers, machine-readable |
| [p1-runs.md](p1-runs.md) | each agentic-decode (P1) run vs stock, paired by request, with 95 % CIs |
| [clean-clone-check.md](clean-clone-check.md) | cold start from a fresh clone, functional checks, KL against the tuned configuration |
| [launch-check.md](launch-check.md) | mechanical comparison of `serve.sh` with the launches that were measured |
| [screening.md](screening.md) | the effect of each change when it was screened |
| [quality-battery.md](quality-battery.md) | quality battery of the output-changing steps, KL gates of the others |
| [fin3c-quality-investigation.md](fin3c-quality-investigation.md) | `fastest` vs `fast`: the dedicated quality investigation |

Configuration names used during tuning: `fin-g` = `lossless`, `gsf-f2` = `fast`, `fin3c` = `fastest`, `s0` = stock.
`fin`, `fin-w8`, `fin2` and `dsm-c2` are intermediate steps with the same outputs as, respectively, `lossless`
without the fused gate, `fast` without the far-block path, `fast`, and `fastest`.

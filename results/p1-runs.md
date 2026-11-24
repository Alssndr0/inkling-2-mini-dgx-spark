# P1 runs vs stock (lab compare.py output)

Run 1 (first pass from the clean clone; the fastest run compiled the overlay kernels while it measured):

```
p1-rc-fastest/rc-fastest vs p1-s0/s0: 177 requests, 25 sessions
  decode tok/s      A   62.33   B   80.82   B/A +29.68%  95% CI [+25.30%, +33.84%]
  tokens per step   A   4.084   B   3.945   B/A -3.38%  95% CI [-6.73%, -0.26%]
  accepted per pos  A 0.806 0.625 0.492 0.390   B 0.795 0.610 0.469 0.366
  CJK replies       A    0   B    0
  empty replies     A    0   B    0
  cut at the cap    A   14   B   20
  tool-call args valid (uncut replies)  A 169/169   B 168/168
{"all": {"tok_s_ratio": 1.2967510567395504, "tok_s_ci": [1.2530353752763799, 1.3384172668656806], "step_ratio": 0.9661873423287313, "step_ci": [0.9327140483688651, 0.9973997482558822]}}
p1-rc-fast/rc-fast vs p1-s0/s0: 177 requests, 25 sessions
  decode tok/s      A   62.33   B   83.61   B/A +34.16%  95% CI [+30.13%, +38.83%]
  tokens per step   A   4.084   B   4.080   B/A -0.08%  95% CI [-3.09%, +3.34%]
  accepted per pos  A 0.806 0.625 0.492 0.390   B 0.801 0.629 0.489 0.387
  CJK replies       A    0   B    0
  empty replies     A    0   B    0
  cut at the cap    A   14   B   18
  tool-call args valid (uncut replies)  A 169/169   B 162/162
{"all": {"tok_s_ratio": 1.3415697067287498, "tok_s_ci": [1.3012915213550755, 1.3882550836760847], "step_ratio": 0.9992413770212629, "step_ci": [0.9691262230544859, 1.0334231036900259]}}
p1-rc-lossless/rc-lossless vs p1-s0/s0: 177 requests, 25 sessions
  decode tok/s      A   62.33   B   74.50   B/A +19.54%  95% CI [+15.72%, +24.02%]
  tokens per step   A   4.084   B   4.131   B/A +1.16%  95% CI [-2.01%, +4.69%]
  accepted per pos  A 0.806 0.625 0.492 0.390   B 0.809 0.636 0.497 0.394
  CJK replies       A    0   B    0
  empty replies     A    0   B    0
  cut at the cap    A   14   B   22
  tool-call args valid (uncut replies)  A 169/169   B 167/167
{"all": {"tok_s_ratio": 1.1953995963480282, "tok_s_ci": [1.1571884322751727, 1.2401682770031255], "step_ratio": 1.0115698934485484, "step_ci": [0.9799330291787297, 1.0468699977553042]}}
```

Run 2 (warm pass) for fast and fastest:

```
p1-rc-fastest-2/rc-fastest-2 vs p1-s0/s0: 177 requests, 25 sessions
  decode tok/s      A   62.33   B   82.09   B/A +31.70%  95% CI [+27.22%, +35.92%]
  tokens per step   A   4.084   B   3.945   B/A -3.38%  95% CI [-6.73%, -0.26%]
  accepted per pos  A 0.806 0.625 0.492 0.390   B 0.795 0.610 0.469 0.366
  CJK replies       A    0   B    0
  empty replies     A    0   B    0
  cut at the cap    A   14   B   20
  tool-call args valid (uncut replies)  A 169/169   B 168/168
{"all": {"tok_s_ratio": 1.3170382451062357, "tok_s_ci": [1.272151364621126, 1.3592104048569282], "step_ratio": 0.9661873423287313, "step_ci": [0.9327140483688651, 0.9973997482558822]}}
p1-rc-fast-2/rc-fast-2 vs p1-s0/s0: 177 requests, 25 sessions
  decode tok/s      A   62.33   B   83.58   B/A +34.10%  95% CI [+30.10%, +38.74%]
  tokens per step   A   4.084   B   4.080   B/A -0.08%  95% CI [-3.09%, +3.34%]
  accepted per pos  A 0.806 0.625 0.492 0.390   B 0.801 0.629 0.489 0.387
  CJK replies       A    0   B    0
  empty replies     A    0   B    0
  cut at the cap    A   14   B   18
  tool-call args valid (uncut replies)  A 169/169   B 162/162
{"all": {"tok_s_ratio": 1.340964691097315, "tok_s_ci": [1.3010276000118384, 1.387383980554252], "step_ratio": 0.9992413770212629, "step_ci": [0.9691262230544859, 1.0334231036900259]}}
```

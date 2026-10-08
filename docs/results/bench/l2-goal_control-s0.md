# Inference speed: greedy vs PUCT

- env: `textworld` L2-goal
- checkpoint: `runs/ablation/selfplay_textworld_v2/L2-goal/control-s0/checkpoints/final`
- device: `cuda:0`
- games per method: 30 (eval split), warm-up 2
- PUCT: {'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

| method | success | reward | moves | success moves | s/game | ms/move | p50 | p95 | × greedy | model calls/move | rows/move | model % | env % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.70 | 0.547 | 9.9 | 5.6 | 0.20 | 13 | 13 | 16 | 1.0 | 1.0 | 2.0 | 66 | 34 |
| puct16 | 0.97 | 0.745 | 6.6 | 6.1 | 1.35 | 206 | 158 | 505 | 15.6 | 3.9 | 12.1 | 21 | 78 |
| puct64 | 1.00 | 0.790 | 5.2 | 5.2 | 1.30 | 250 | 103 | 1086 | 19.0 | 6.5 | 17.8 | 24 | 75 |

moves: all games (a failed game counts the moves it used); success moves: successful games only. model % / env %: share of the game time spent in Laya calls (tokenisation + forward) / in environment steps; the rest is search bookkeeping. × greedy: latency per move relative to greedy.

# phase0-algebra-tiny-from-scratch

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.730 | 7.420 | 0.578 |
| uniform-gumbel16 | 1.000 | 0.735 | 7.210 | 0.703 |
| rollout-puct16 | 0.980 | 0.706 | 7.940 | 1.427 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 9.970 | 0.014 | 0.020 | 9.910 | 0.042 | 0.060 | 9.770 | 0.073 | 0.100 |
| 1 | selfplay | 10.000 | 0.000 | 0.000 | 7.780 | 0.602 | 0.820 | 8.040 | 0.542 | 0.740 |
| 2 | selfplay | 10.000 | 0.000 | 0.000 | 7.950 | 0.597 | 0.820 | 8.110 | 0.567 | 0.780 |
| 3 | selfplay | 9.320 | 0.372 | 0.530 | 7.380 | 0.698 | 0.950 | 7.640 | 0.692 | 0.950 |
| 4 | selfplay | 9.260 | 0.528 | 0.760 | 7.250 | 0.708 | 0.960 | 7.170 | 0.716 | 0.970 |
| 5 | selfplay | 8.650 | 0.602 | 0.850 | 7.490 | 0.715 | 0.980 | 7.140 | 0.717 | 0.970 |
| 6 | selfplay | 8.080 | 0.714 | 1.000 | 8.730 | 0.566 | 0.800 | 7.560 | 0.694 | 0.950 |
| 7 | selfplay | 8.340 | 0.642 | 0.900 | 8.030 | 0.696 | 0.970 | 7.300 | 0.700 | 0.950 |
| 8 | selfplay | 8.350 | 0.608 | 0.850 | 7.950 | 0.671 | 0.930 | 7.020 | 0.740 | 1.000 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.219 | 9.734 | 623 | 38.624 | 314.879 |
| 2 | 64 | 0.328 | 9.344 | 598 | 36.171 | 314.452 |
| 3 | 64 | 0.641 | 8.656 | 554 | 32.696 | 309.826 |
| 4 | 64 | 0.734 | 8.203 | 525 | 33.404 | 286.796 |
| 5 | 64 | 0.953 | 7.531 | 482 | 26.686 | 300.009 |
| 6 | 64 | 0.891 | 7.797 | 499 | 27.900 | 290.031 |
| 7 | 64 | 0.906 | 8.234 | 527 | 31.017 | 278.044 |
| 8 | 64 | 0.922 | 8.359 | 535 | 30.755 | 287.299 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 1 | selfplay | 623 | 1.362 | 0.490 | 0.274 | 0.063 | 5.000 | 1.046 |
| 2 | selfplay | 1221 | 1.364 | 0.445 | 0.377 | 0.080 | 1.000 | 0.830 |
| 3 | selfplay | 1775 | 1.445 | 0.536 | 0.362 | 0.096 | 2.443 | 0.830 |
| 4 | selfplay | 2300 | 1.333 | 0.573 | 0.374 | 0.109 | 1.138 | 1.460 |
| 5 | selfplay | 2782 | 1.331 | 0.617 | 0.482 | 0.106 | 2.096 | 0.804 |
| 6 | selfplay | 3281 | 1.301 | 0.641 | 0.500 | 0.106 | 1.640 | 0.738 |
| 7 | selfplay | 3808 | 1.276 | 0.646 | 0.526 | 0.097 | 1.307 | 0.757 |
| 8 | selfplay | 4343 | 1.261 | 0.647 | 0.578 | 0.103 | 1.364 | 1.347 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.542 -> 0.740 | PASS |
| puct16 beats greedy (no search) | 0.740 vs 0.608 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.740 vs 0.730 | PASS |

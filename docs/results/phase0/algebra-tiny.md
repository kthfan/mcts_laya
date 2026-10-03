# phase0-algebra-tiny

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.730 | 7.420 | 0.651 |
| uniform-gumbel16 | 1.000 | 0.735 | 7.210 | 0.711 |
| rollout-puct16 | 0.980 | 0.706 | 7.940 | 1.448 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 9.970 | 0.014 | 0.020 | 9.910 | 0.042 | 0.060 | 9.770 | 0.073 | 0.100 |
| 0 | warmstart | 8.210 | 0.598 | 0.830 | 8.090 | 0.515 | 0.700 | 7.070 | 0.732 | 0.990 |
| 1 | selfplay | 8.470 | 0.639 | 0.900 | 7.770 | 0.628 | 0.860 | 6.620 | 0.750 | 1.000 |
| 2 | selfplay | 8.000 | 0.703 | 0.980 | 7.550 | 0.614 | 0.830 | 6.690 | 0.748 | 1.000 |
| 3 | selfplay | 7.460 | 0.710 | 0.970 | 7.580 | 0.622 | 0.840 | 6.620 | 0.750 | 1.000 |
| 4 | selfplay | 7.890 | 0.720 | 1.000 | 7.130 | 0.671 | 0.900 | 6.760 | 0.746 | 1.000 |
| 5 | selfplay | 7.910 | 0.719 | 1.000 | 6.760 | 0.747 | 1.000 | 6.710 | 0.748 | 1.000 |
| 6 | selfplay | 7.930 | 0.719 | 1.000 | 6.520 | 0.753 | 1.000 | 6.600 | 0.751 | 1.000 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.844 | 7.891 | 505 | 30.989 | 286.551 |
| 2 | 64 | 0.750 | 7.781 | 498 | 30.416 | 286.037 |
| 3 | 64 | 1.000 | 6.688 | 428 | 24.466 | 282.274 |
| 4 | 64 | 1.000 | 6.672 | 427 | 22.581 | 310.352 |
| 5 | 64 | 1.000 | 6.266 | 401 | 22.170 | 301.847 |
| 6 | 64 | 0.969 | 6.797 | 435 | 23.964 | 301.123 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 1101 | 1.209 | 0.595 | 0.700 | 0.010 | 1.122 | 1.105 |
| 1 | selfplay | 1055 | 1.171 | 0.647 | 0.752 | 0.060 | 1.661 | 1.119 |
| 2 | selfplay | 1553 | 1.208 | 0.665 | 0.845 | 0.087 | 1.211 | 0.943 |
| 3 | selfplay | 1981 | 1.185 | 0.649 | 0.833 | 0.045 | 0.813 | 0.732 |
| 4 | selfplay | 2408 | 1.201 | 0.639 | 0.879 | 0.044 | 1.157 | 0.637 |
| 5 | selfplay | 2809 | 1.191 | 0.633 | 0.868 | 0.051 | 0.965 | 1.104 |
| 6 | selfplay | 3244 | 1.191 | 0.626 | 0.877 | 0.054 | 1.398 | 1.230 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.732 -> 0.751 | PASS |
| puct16 beats greedy (no search) | 0.751 vs 0.719 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.751 vs 0.730 | PASS |

# phase0-algebra-tiny

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.730 | 7.420 | 0.596 |
| uniform-gumbel16 | 1.000 | 0.735 | 7.210 | 0.679 |
| rollout-puct16 | 0.980 | 0.706 | 7.940 | 1.610 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 10.000 | 0.000 | 0.000 | 9.680 | 0.088 | 0.120 | 9.520 | 0.119 | 0.160 |
| 0 | warmstart | 8.230 | 0.378 | 0.500 | 8.970 | 0.260 | 0.350 | 6.870 | 0.712 | 0.950 |
| 1 | selfplay | 8.220 | 0.545 | 0.750 | 7.410 | 0.598 | 0.800 | 6.300 | 0.752 | 0.990 |
| 2 | selfplay | 7.900 | 0.433 | 0.570 | 6.740 | 0.716 | 0.950 | 6.350 | 0.751 | 0.990 |
| 3 | selfplay | 8.310 | 0.529 | 0.730 | 7.160 | 0.657 | 0.880 | 6.800 | 0.746 | 1.000 |
| 4 | selfplay | 7.920 | 0.492 | 0.660 | 6.310 | 0.699 | 0.910 | 5.990 | 0.767 | 1.000 |
| 5 | selfplay | 7.460 | 0.710 | 0.970 | 6.130 | 0.731 | 0.950 | 5.930 | 0.769 | 1.000 |
| 6 | selfplay | 7.880 | 0.440 | 0.580 | 6.280 | 0.707 | 0.920 | 5.940 | 0.756 | 0.980 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.781 | 8.094 | 518 | 36.841 | 249.018 |
| 2 | 64 | 0.938 | 6.672 | 427 | 29.490 | 234.859 |
| 3 | 64 | 0.938 | 6.609 | 423 | 28.046 | 235.114 |
| 4 | 64 | 0.953 | 6.984 | 447 | 32.493 | 232.542 |
| 5 | 64 | 1.000 | 6.484 | 415 | 28.696 | 227.976 |
| 6 | 64 | 0.969 | 6.594 | 422 | 28.318 | 233.277 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 7659 | 1.126 | 0.601 | 0.805 | 0.028 | 1.416 | 1.042 |
| 1 | selfplay | 4347 | 1.117 | 0.620 | 0.810 | 0.034 | 0.735 | 0.519 |
| 2 | selfplay | 4774 | 1.090 | 0.620 | 0.858 | 0.034 | 0.775 | 0.887 |
| 3 | selfplay | 5197 | 1.077 | 0.620 | 0.812 | 0.040 | 1.327 | 1.088 |
| 4 | selfplay | 5644 | 1.085 | 0.619 | 0.840 | 0.038 | 1.162 | 1.026 |
| 5 | selfplay | 6059 | 1.080 | 0.617 | 0.845 | 0.035 | 1.155 | 1.027 |
| 6 | selfplay | 6481 | 1.075 | 0.615 | 0.863 | 0.034 | 0.927 | 0.884 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.712 -> 0.756 | PASS |
| puct16 beats greedy (no search) | 0.756 vs 0.440 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.756 vs 0.730 | PASS |

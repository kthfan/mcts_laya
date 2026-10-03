# phase0-algebra-tiny

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.730 | 7.420 | 0.633 |
| uniform-gumbel16 | 1.000 | 0.735 | 7.210 | 0.716 |
| rollout-puct16 | 0.980 | 0.706 | 7.940 | 1.505 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 9.970 | 0.014 | 0.020 | 9.910 | 0.042 | 0.060 | 9.770 | 0.073 | 0.100 |
| 0 | warmstart | 5.870 | 0.764 | 0.990 | 5.630 | 0.777 | 1.000 | 5.690 | 0.769 | 0.990 |
| 1 | selfplay | 6.250 | 0.707 | 0.920 | 5.800 | 0.759 | 0.980 | 5.740 | 0.774 | 1.000 |
| 2 | selfplay | 6.190 | 0.742 | 0.970 | 5.910 | 0.743 | 0.960 | 6.070 | 0.719 | 0.930 |
| 3 | selfplay | 5.840 | 0.771 | 1.000 | 5.800 | 0.752 | 0.970 | 5.710 | 0.775 | 1.000 |
| 4 | selfplay | 5.870 | 0.750 | 0.970 | 5.700 | 0.775 | 1.000 | 5.760 | 0.773 | 1.000 |
| 5 | selfplay | 6.060 | 0.752 | 0.980 | 5.710 | 0.774 | 1.000 | 5.870 | 0.770 | 1.000 |
| 6 | selfplay | 5.670 | 0.776 | 1.000 | 5.610 | 0.777 | 1.000 | 5.820 | 0.772 | 1.000 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.984 | 6.219 | 398 | 21.876 | 253.152 |
| 2 | 64 | 1.000 | 5.984 | 383 | 20.100 | 241.393 |
| 3 | 64 | 0.891 | 6.547 | 419 | 22.353 | 257.591 |
| 4 | 64 | 1.000 | 6.047 | 387 | 19.518 | 262.213 |
| 5 | 64 | 0.984 | 5.953 | 381 | 18.390 | 263.838 |
| 6 | 64 | 0.984 | 6.078 | 389 | 19.729 | 259.218 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 7659 | 1.051 | 0.600 | 0.907 | 0.029 | 1.250 | 1.287 |
| 1 | selfplay | 4227 | 0.829 | 0.597 | 0.900 | 0.022 | 0.750 | 0.716 |
| 2 | selfplay | 4610 | 0.791 | 0.592 | 0.902 | 0.023 | 0.923 | 0.908 |
| 3 | selfplay | 5029 | 0.779 | 0.599 | 0.877 | 0.028 | 1.211 | 0.956 |
| 4 | selfplay | 5416 | 0.793 | 0.596 | 0.915 | 0.020 | 0.910 | 0.787 |
| 5 | selfplay | 5797 | 0.806 | 0.595 | 0.912 | 0.023 | 1.019 | 1.070 |
| 6 | selfplay | 6186 | 0.776 | 0.596 | 0.930 | 0.024 | 1.132 | 1.038 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.769 -> 0.772 | PASS |
| puct16 beats greedy (no search) | 0.772 vs 0.776 | FAIL |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.772 vs 0.730 | PASS |

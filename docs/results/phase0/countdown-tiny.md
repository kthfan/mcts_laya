# phase0-countdown-tiny

- env: `countdown` {'n_numbers': 4, 'target_range': [10, 200], 'min_solution_steps': 2, 'action_detail': 'outcome'}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.300 | 0.300 | 2.850 | 0.122 |
| uniform-gumbel16 | 0.330 | 0.330 | 2.830 | 0.126 |
| rollout-puct16 | 0.430 | 0.430 | 2.870 | 0.169 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 3.000 | 0.030 | 0.030 | 2.920 | 0.190 | 0.190 | 2.930 | 0.220 | 0.220 |
| 0 | warmstart | 2.890 | 0.180 | 0.180 | 2.810 | 0.320 | 0.320 | 2.860 | 0.370 | 0.370 |
| 1 | selfplay | 2.880 | 0.150 | 0.150 | 2.810 | 0.320 | 0.320 | 2.860 | 0.280 | 0.280 |
| 2 | selfplay | 2.950 | 0.110 | 0.110 | 2.750 | 0.340 | 0.340 | 2.870 | 0.270 | 0.270 |
| 3 | selfplay | 2.910 | 0.150 | 0.150 | 2.710 | 0.440 | 0.440 | 2.870 | 0.300 | 0.300 |
| 4 | selfplay | 2.930 | 0.150 | 0.150 | 2.830 | 0.380 | 0.380 | 2.920 | 0.350 | 0.350 |
| 5 | selfplay | 2.930 | 0.120 | 0.120 | 2.780 | 0.370 | 0.370 | 2.860 | 0.290 | 0.290 |
| 6 | selfplay | 2.890 | 0.150 | 0.150 | 2.730 | 0.380 | 0.380 | 2.820 | 0.350 | 0.350 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.453 | 2.688 | 172 | 13.263 | 194.528 |
| 2 | 64 | 0.344 | 2.859 | 183 | 13.903 | 193.774 |
| 3 | 64 | 0.375 | 2.844 | 182 | 13.692 | 206.395 |
| 4 | 64 | 0.359 | 2.812 | 180 | 13.878 | 186.767 |
| 5 | 64 | 0.344 | 2.828 | 181 | 14.685 | 188.220 |
| 6 | 64 | 0.328 | 2.797 | 179 | 14.236 | 182.501 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 2729 | 1.838 | 0.437 | 0.636 | 0.160 | 1.569 | 1.491 |
| 1 | selfplay | 1536 | 1.761 | 0.545 | 0.569 | 0.167 | 0.717 | 0.633 |
| 2 | selfplay | 1719 | 1.752 | 0.589 | 0.515 | 0.193 | 0.500 | 1.091 |
| 3 | selfplay | 1901 | 1.706 | 0.629 | 0.474 | 0.239 | 1.429 | 0.831 |
| 4 | selfplay | 2081 | 1.704 | 0.629 | 0.562 | 0.225 | 0.854 | 1.151 |
| 5 | selfplay | 2262 | 1.683 | 0.675 | 0.549 | 0.223 | 1.638 | 0.673 |
| 6 | selfplay | 2441 | 1.695 | 0.671 | 0.516 | 0.213 | 0.979 | 0.778 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.370 -> 0.350 | FAIL |
| puct16 beats greedy (no search) | 0.350 vs 0.150 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.350 vs 0.300 | PASS |

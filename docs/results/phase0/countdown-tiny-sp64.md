# phase0-countdown-tiny-selfplay64

- env: `countdown` {'n_numbers': 4, 'target_range': [10, 200], 'min_solution_steps': 2, 'action_detail': 'outcome'}
- checkpoint: `models/tiny`
- search: `puct` {'num_simulations': 64, 'batch_size': 8, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.300 | 0.300 | 2.850 | 0.116 |
| uniform-gumbel16 | 0.330 | 0.330 | 2.830 | 0.123 |
| rollout-puct16 | 0.430 | 0.430 | 2.870 | 0.176 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | gumbel16.moves | gumbel16.reward | gumbel16.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | initial | 3.000 | 0.030 | 0.030 | 2.920 | 0.190 | 0.190 | 2.930 | 0.220 | 0.220 |
| 0 | warmstart | 2.990 | 0.020 | 0.020 | 2.880 | 0.210 | 0.210 | 2.920 | 0.190 | 0.190 |
| 1 | selfplay | 2.990 | 0.040 | 0.040 | 2.910 | 0.190 | 0.190 | 2.890 | 0.260 | 0.260 |
| 2 | selfplay | 3.000 | 0.030 | 0.030 | 2.930 | 0.190 | 0.190 | 2.940 | 0.170 | 0.170 |
| 3 | selfplay | 2.990 | 0.060 | 0.060 | 2.880 | 0.230 | 0.230 | 2.900 | 0.240 | 0.240 |
| 4 | selfplay | 3.000 | 0.060 | 0.060 | 2.870 | 0.280 | 0.280 | 2.850 | 0.260 | 0.260 |
| 5 | selfplay | 2.990 | 0.040 | 0.040 | 2.910 | 0.250 | 0.250 | 2.920 | 0.210 | 0.210 |
| 6 | selfplay | 2.990 | 0.050 | 0.050 | 2.920 | 0.160 | 0.160 | 2.950 | 0.140 | 0.140 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.719 | 2.688 | 172 | 25.218 | 221.986 |
| 2 | 64 | 0.625 | 2.734 | 175 | 25.785 | 214.311 |
| 3 | 64 | 0.625 | 2.812 | 180 | 22.564 | 226.641 |
| 4 | 64 | 0.656 | 2.703 | 173 | 22.660 | 234.775 |
| 5 | 64 | 0.719 | 2.750 | 176 | 22.777 | 227.157 |
| 6 | 64 | 0.609 | 2.781 | 178 | 21.686 | 244.675 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 2729 | 2.097 | 0.450 | 0.507 | 0.144 | 0.965 | 1.291 |
| 1 | selfplay | 1536 | 2.111 | 0.475 | 0.425 | 0.154 | 0.970 | 1.043 |
| 2 | selfplay | 1711 | 2.105 | 0.507 | 0.398 | 0.164 | 1.000 | 0.674 |
| 3 | selfplay | 1891 | 2.101 | 0.541 | 0.328 | 0.183 | 1.000 | 0.961 |
| 4 | selfplay | 2064 | 2.106 | 0.540 | 0.335 | 0.217 | 1.000 | 1.191 |
| 5 | selfplay | 2240 | 2.109 | 0.565 | 0.321 | 0.188 | 5.000 | 0.731 |
| 6 | selfplay | 2418 | 2.109 | 0.591 | 0.315 | 0.208 | 1.530 | 1.639 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.190 -> 0.140 | FAIL |
| puct16 beats greedy (no search) | 0.140 vs 0.050 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.140 vs 0.300 | FAIL |

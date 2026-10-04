# phase1a-textworld-l2-laya-cpu

- env: `textworld` {'game_dir': 'data/textworld', 'level': 'L2', 'gamma': 0.9, 'success_floor': 0.5, 'drop_commands': ['look', 'inventory', 'examine'], 'history_len': 6}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.033 | 0.028 | 14.633 | 381.280 |
| rollout-puct16 | 0.233 | 0.173 | 13.167 | 833.160 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 14.500 | 0.069 | 0.100 | 14.200 | 0.175 | 0.267 |
| 0 | warmstart | 6.800 | 0.694 | 0.900 | 6.533 | 0.761 | 1.000 |
| 1 | selfplay | 6.967 | 0.656 | 0.833 | 6.367 | 0.747 | 0.967 |
| 2 | selfplay | 6.700 | 0.680 | 0.867 | 5.767 | 0.760 | 0.967 |
| 3 | selfplay | 6.700 | 0.662 | 0.833 | 6.367 | 0.708 | 0.900 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 12 | 0.917 | 7.083 | 85 | 360.966 | 3.275 |
| 2 | 12 | 1.000 | 5.333 | 64 | 139.820 | 3.905 |
| 3 | 12 | 1.000 | 4.917 | 59 | 172.779 | 3.797 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 461 | 0.593 | 0.569 | 0.870 | 0.001 | 2.639 | 1.016 |
| 1 | selfplay | 315 | 0.508 | 0.571 | 0.935 | 0.035 | 1.359 | 2.115 |
| 2 | selfplay | 379 | 0.394 | 0.597 | 0.892 | 0.024 | 1.437 | 1.691 |
| 3 | selfplay | 438 | 0.494 | 0.574 | 0.953 | 0.001 | 0.948 | 0.980 |

## Milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.761 -> 0.708 | FAIL |
| puct16 beats greedy (no search) | 0.708 vs 0.662 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.708 vs 0.028 | PASS |

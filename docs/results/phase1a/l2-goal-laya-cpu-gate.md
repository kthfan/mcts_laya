# phase1a-textworld-l2-goal-laya-cpu

- env: `textworld` {'game_dir': 'data/textworld', 'level': 'L2-goal', 'gamma': 0.9, 'success_floor': 0.5, 'drop_commands': ['look', 'inventory', 'examine'], 'history_len': 6}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.050 | 0.041 | 19.200 | 430.318 |
| rollout-puct16 | 0.400 | 0.286 | 15.450 | 855.545 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 18.800 | 0.072 | 0.100 | 17.550 | 0.146 | 0.200 |
| 0 | warmstart | 7.950 | 0.661 | 0.850 | 5.850 | 0.754 | 0.950 |
| 1 | selfplay | 10.700 | 0.509 | 0.650 | 9.400 | 0.611 | 0.800 |
| 2 | selfplay | 12.700 | 0.393 | 0.500 | 10.950 | 0.534 | 0.700 |
| 3 | selfplay | 9.050 | 0.589 | 0.750 | 7.650 | 0.694 | 0.900 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 12 | 0.667 | 9.917 | 119 | 483.762 | 4.039 |
| 2 | 12 | 0.750 | 9.083 | 109 | 463.076 | 3.762 |
| 3 | 12 | 0.917 | 8.250 | 99 | 353.016 | 3.977 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 606 | 0.804 | 0.572 | 0.867 | 0.002 | 2.353 | 1.126 |
| 1 | selfplay | 725 | 0.345 | 0.554 | 0.833 | 0.002 | 2.503 | 0.936 |
| 2 | selfplay | 834 | 0.511 | 0.553 | 0.892 | 0.008 | 2.168 | 1.103 |
| 3 | selfplay | 933 | 0.369 | 0.546 | 0.903 | 0.025 | 1.855 | 1.547 |

## Milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.754 -> 0.694 | FAIL |
| puct16 beats greedy (no search) | 0.694 vs 0.589 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.694 vs 0.041 | PASS |

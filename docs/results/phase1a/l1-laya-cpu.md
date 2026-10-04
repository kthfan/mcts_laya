# phase1a-textworld-l1-laya-cpu

- env: `textworld` {'game_dir': 'data/textworld', 'level': 'L1', 'gamma': 0.9, 'success_floor': 0.5, 'drop_commands': ['look', 'inventory', 'examine'], 'history_len': 6}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.300 | 0.245 | 8.300 | 128.242 |
| rollout-puct16 | 0.733 | 0.595 | 6.067 | 199.444 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 8.633 | 0.193 | 0.233 | 8.467 | 0.375 | 0.500 |
| 0 | warmstart | 3.667 | 0.779 | 0.900 | 3.200 | 0.837 | 0.967 |
| 1 | selfplay | 3.233 | 0.836 | 0.967 | 3.233 | 0.836 | 0.967 |
| 2 | selfplay | 2.967 | 0.866 | 1.000 | 2.967 | 0.866 | 1.000 |
| 3 | selfplay | 3.000 | 0.865 | 1.000 | 2.967 | 0.866 | 1.000 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 12 | 1.000 | 3.000 | 36 | 48.642 | 5.098 |
| 2 | 12 | 1.000 | 3.000 | 36 | 53.766 | 5.133 |
| 3 | 12 | 0.833 | 4.167 | 50 | 131.755 | 4.007 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 255 | 0.486 | 0.468 | 1.000 | 0.001 | 0.500 | 1.011 |
| 1 | selfplay | 163 | 0.150 | 0.440 | 0.938 | 0.001 | 1.604 | 1.017 |
| 2 | selfplay | 199 | 0.141 | 0.434 | 1.000 | 0.000 | 1.048 | 1.184 |
| 3 | selfplay | 249 | 0.108 | 0.415 | 0.958 | 0.001 | 1.074 | 0.996 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.837 -> 0.866 | PASS |
| puct16 beats greedy (no search) | 0.866 vs 0.865 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.866 vs 0.245 | PASS |

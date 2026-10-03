# phase0-countdown-laya-cpu-smoke

- env: `countdown` {'n_numbers': 4, 'target_range': [10, 200], 'min_solution_steps': 2, 'action_detail': 'outcome'}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 0.300 | 0.300 | 2.900 | 0.030 |
| uniform-gumbel16 | 0.333 | 0.333 | 2.900 | 0.034 |
| rollout-puct16 | 0.567 | 0.567 | 2.867 | 0.038 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 2.967 | 0.100 | 0.100 | 2.967 | 0.167 | 0.167 |
| 0 | warmstart | 2.700 | 0.700 | 0.700 | 2.633 | 0.833 | 0.833 |
| 1 | selfplay | 2.767 | 0.700 | 0.700 | 2.633 | 0.933 | 0.933 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 16 | 0.812 | 2.750 | 44 | 75.626 | 6.797 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 403 | 1.698 | 0.538 | 0.775 | 0.168 | 1.411 | 2.503 |
| 1 | selfplay | 245 | 1.407 | 0.648 | 0.708 | 0.170 | 1.646 | 0.500 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.833 -> 0.933 | PASS |
| puct16 beats greedy (no search) | 0.933 vs 0.700 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.933 vs 0.300 | PASS |

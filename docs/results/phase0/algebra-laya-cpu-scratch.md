# phase0-algebra-laya-cpu-from-scratch

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.727 | 7.533 | 0.167 |
| uniform-gumbel16 | 1.000 | 0.731 | 7.333 | 0.199 |
| rollout-puct16 | 1.000 | 0.720 | 7.933 | 0.394 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 9.933 | 0.024 | 0.033 | 9.500 | 0.348 | 0.500 |
| 1 | selfplay | 9.367 | 0.193 | 0.267 | 7.767 | 0.677 | 0.933 |
| 2 | selfplay | 7.867 | 0.719 | 1.000 | 7.633 | 0.702 | 0.967 |
| 3 | selfplay | 8.067 | 0.714 | 1.000 | 7.500 | 0.728 | 1.000 |
| 4 | selfplay | 8.067 | 0.714 | 1.000 | 7.400 | 0.708 | 0.967 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 16 | 0.250 | 9.562 | 153 | 243.996 | 11.566 |
| 2 | 16 | 0.562 | 8.750 | 140 | 208.457 | 11.523 |
| 3 | 16 | 1.000 | 7.125 | 114 | 148.505 | 11.474 |
| 4 | 16 | 1.000 | 7.688 | 123 | 147.973 | 12.583 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 1 | selfplay | 153 | 0.863 | 0.533 | 0.333 | 0.054 | 2.661 | 0.994 |
| 2 | selfplay | 293 | 0.970 | 0.593 | 0.655 | 0.098 | 1.054 | 0.811 |
| 3 | selfplay | 407 | 1.170 | 0.592 | 0.725 | 0.047 | 1.616 | 1.299 |
| 4 | selfplay | 530 | 1.144 | 0.544 | 0.698 | 0.078 | 1.821 | 4.361 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.677 -> 0.708 | PASS |
| puct16 beats greedy (no search) | 0.708 vs 0.714 | FAIL |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.708 vs 0.727 | FAIL |

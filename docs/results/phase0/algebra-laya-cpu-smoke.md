# phase0-algebra-laya-cpu-smoke

- env: `algebra` {'templates': ['bracket', 'fraction', 'two_brackets', 'two_fractions'], 'max_steps': 10, 'gamma': 0.9, 'success_floor': 0.5}
- checkpoint: `models/laya/multilingual`
- search: `puct` {'num_simulations': 16, 'batch_size': 4, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_fraction': 0.25}

## Baselines (no Laya)

| label | success | reward | moves | seconds |
|---|---|---|---|---|
| uniform-puct16 | 1.000 | 0.727 | 7.533 | 0.195 |
| uniform-gumbel16 | 1.000 | 0.731 | 7.333 | 0.196 |
| rollout-puct16 | 1.000 | 0.720 | 7.933 | 0.411 |

## Learning curve

| iteration | stage | greedy.moves | greedy.reward | greedy.success | puct16.moves | puct16.reward | puct16.success |
|---|---|---|---|---|---|---|---|
| 0 | initial | 9.933 | 0.024 | 0.033 | 9.500 | 0.348 | 0.500 |
| 0 | warmstart | 5.667 | 0.776 | 1.000 | 5.667 | 0.776 | 1.000 |
| 1 | selfplay | 5.833 | 0.771 | 1.000 | 5.733 | 0.774 | 1.000 |
| 2 | selfplay | 5.700 | 0.775 | 1.000 | 5.667 | 0.776 | 1.000 |
| 3 | selfplay | 5.667 | 0.776 | 1.000 | 5.667 | 0.776 | 1.000 |
| 4 | selfplay | 5.667 | 0.776 | 1.000 | 5.633 | 0.777 | 1.000 |

## Self-play

| iteration | episodes | success | moves | samples | seconds | rows_per_second |
|---|---|---|---|---|---|---|
| 1 | 16 | 1.000 | 5.688 | 91 | 93.521 | 11.014 |
| 2 | 16 | 1.000 | 5.875 | 94 | 103.506 | 10.782 |
| 3 | 16 | 1.000 | 5.625 | 90 | 89.619 | 11.091 |
| 4 | 16 | 1.000 | 5.938 | 95 | 118.032 | 10.794 |

## Training

| iteration | stage | samples | policy_ce | value_ce | heldout_policy_top1 | heldout_value_brier | temperature_choice | temperature_noul |
|---|---|---|---|---|---|---|---|---|
| 0 | warmstart | 1101 | 0.694 | 0.584 | 0.973 | 0.007 | 1.322 | 1.098 |
| 1 | selfplay | 641 | 0.548 | 0.559 | 1.000 | 0.012 | 1.324 | 1.161 |
| 2 | selfplay | 735 | 0.540 | 0.554 | 0.945 | 0.014 | 1.163 | 1.157 |
| 3 | selfplay | 825 | 0.555 | 0.548 | 1.000 | 0.000 | 1.134 | 1.066 |
| 4 | selfplay | 920 | 0.575 | 0.549 | 0.967 | 0.005 | 0.959 | 1.086 |

## Phase 0 milestone checks

| check | value | result |
|---|---|---|
| puct16.reward improves over self-play iterations | 0.776 -> 0.777 | PASS |
| puct16 beats greedy (no search) | 0.777 vs 0.776 | PASS |
| puct16 beats uniform-puct16 (same budget, no Laya) | 0.777 vs 0.727 | PASS |

# DROGO

This repository contains the code for DROGO: Default Representation Objective via Graph Optimization in Reinforcement Learning.

## Requirements

To install requirements:

```setup
conda create -n drogo python=3.12
conda activate drogo
pip install -r requirements.txt
```

## Experiments

### Datasets

To generate datasets for DROGO:
```
python -m drogo.function_approximation.generate_dataset --env fourrooms_2 --num_transitions 200000
```

To generate datasets for loss functions other than DROGO (e.g., $\ell_\text{GDO}$):
```
python -m drogo.function_approximation.generate_dataset_quad_penalty --env fourrooms_2 --num_transitions 200000
```

### Training
After generating datasets, we can train neural networks using different loss functions.

DROGO:
```
python -m drogo.function_approximation.eigenlearning_dr --dr_mode drogo --obs_type coordinates
```

For different loss functions, state representations, and hyperparameters, change the arguments.

### Reward Shaping

After learning eigenvectors using the code above, to perform reward shaping:
```
python -m drogo.function_approximation.run_reward_shaping --mode DR --obs_type coordinates
```

Change 'mode' to run baselines

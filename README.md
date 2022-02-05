# UOR-RL Source code

This is the source code of DB-UOR-RL and DF-UOR-RL. You can reproduce the results of our experiments with this.

# Installation 

To run the source code, some python packages are necessary.  If you are using Linux system, you can use the following command to install the packages from conda:

```sh
conda env create -f uor_rl.yaml
```

# Environment

The available environments and the name of the environment parameter are shown in the following table.

| Name of the MuJoCo Environment | Environment ID              |
| ------------------------------ | --------------------------- |
| InvertedPendulum               | SunblazeInvertedPendulum-v0 |
| Ant                            | SunblazeAnt-v0              |
| Humanoid                       | SunblazeHumanoid-v0         |
| Walker2d                       | SunblazeWalker2d-v0         |
| Reacher                        | SunblazeReacher-v0          |
| Hopper                         | SunblazeHopper-v0           |
| Half Cheetah                   | SunblazeHalfCheetah-v0      |

## Training Examples

To train DB-UOR-RL:

```sh
nohup python db_uor_rl.py \
--env_id "SunblazeAnt-v0" \ # The environment ID
--seed 2 \ # The random number seed
--eval_k 2 \ # The robustness degree parameter, which is equal to k+1 for the k in the paper
--block_num 100 \ # The number of blocks that environment parameters be divided, which should be a square number
--total_iters 1000 # The total iterations in training
```

To train DF-UOR-RL:

```sh
nohup python df_uor_rl.py \
--env_id "SunblazeAnt-v0" \ # The environment ID
--seed 2 \ # The random number seed
--eval_k 2 \ # The robustness degree parameter, which is equal to k+1 for the k in the paper
--total_iters 1000 # The total iterations in training
```


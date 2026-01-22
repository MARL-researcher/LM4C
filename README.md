# Code for the paper "From Causal Counterfactuals to Communication-Driven Consensus: A General Framework for Lazy Agent Mitigation in Cooperative MARL"

This repository contains the implementation of the **LM4C** algorithm, which is extended from [EPyMARL](https://github.com/uoe-agents/epymarl).

## 1. Installation & Requirements

### Dependencies
The code is implemented in Python 3.9+ and PyTorch. 
To install the necessary python packages, please run:
```bash
pip install -r requirements.txt
```

### Environment Setup
This codebase supports [Hallway](https://github.com/DiXue98/SMS), Cooperation Navagation and Prey Predator in [MPE](https://github.com/openai/multiagent-particle-envs), and [SMACv2](https://github.com/oxwhirl/smacv2) as described in the paper.

**For Hallway:** The Hallway environment is included in the envs/ directory and requires no external installation.

**For Cooperation Navigation and Prey Predator:** The MPE environment is included in the envs/ directory and requires no external installation.

**For StarCraft II (SMACv2):** This codebase follows the standard installation procedure of the official SMACv2 repository. 
Please refer to [oxwhirl/smacv2](https://github.com/oxwhirl/smacv2) for detailed instructions on installing the StarCraft II client (version 4.10) and downloading the required map files.
*Key Requirement:* Please ensure the `SC2PATH` environment variable is set to your StarCraft II installation directory.

## 2. Project Structure

```text
.
├── src/
│   ├── components/                     # Components including buffer...
│   ├── config/                         # Configuration YAML files
│   │   ├── algs/lm4c.yaml              # LM4C hyperparameters
│   │   ├── envs/                       # Environment-specific settings
│   │   │   ├── hallway/hallway.yaml    # Hallway settings
│   │   │   ├── mpe/                    # MPE settings
│   │   │   └── smac_v2/smacv2_configs/ # SMACv2 settings
│   │   └── default.yaml                # Default config
│   ├── controllers/                    # Multi-Agent Controllers
│   ├── learners/                       # Learning algorithms
│   ├── modules/                        # Neural Network architectures
│   │   ├── agents/                     # Agent networks
│   │   ├── layers/                     # Custom neural layers
│   │   └── mixers/                     # Mixing networks
│   ├── envs/                           # Environment wrappers
│   ├── runners/                        # Episode runners
│   │   ├── episode_runner.py           # Single-thread runner
│   │   └── parallel_runner.py          # Parallel runner
│   ├── run/                            # Experiment runner loop
│   ├── utils/                          # Utilities
│   ├── main.py                         # Entry point
├── results/                            # Directory for logs and saved models
└── requirements.txt                    # Python dependencies
```

## 3. Quick Start
To run experiments on the Hallway environment:
```bash
python src/main.py \
    --alg-config=lm4c \
    --env-config=hallway/hallway \
    --cuda_id=0 \
    --manual_seed=10
```
To run experiments on the Cooperation Navigation and Prey Predator environments:
```bash
# Cooperation Navigation
python src/main.py \
    --alg-config=lm4c \
    --env-config=mpe/cooperation_navigation \
    --cuda_id=0 \
    --manual_seed=10

# Prey Predator
python src/main.py \
    --alg-config=lm4c \
    --env-config=mpe/prey_predator \
    --cuda_id=0 \
    --manual_seed=10
```
To run experiments on the SMACv2 environment:
```bash
# Take map protoss_5_vs_5 as an example, while running other maps can set the specific map name as "--env-config=smac_v2/smacv2_configs/map_name"
python src/main.py \
    --alg-config=lm4c \
    --env-config=smac_v2/smacv2_configs/protoss_5_vs_5 \
    --cuda_id=0 \
    --manual_seed=10
```
The training results will be reported in `results/*`. The training logs (Tensorboard events) will be saved in `results/tb_logs/`. You can visualize the training progress by running:
```bash
# Take environment Hallway as an example
tensorboard --logdir="results/tb_logs/hallway"
```
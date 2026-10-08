# LM4C

Official implementation of LM4C (*Diligence-Aware Cooperative MARL through Causal Influence and Communication-Driven Consensus*).

LM4C is a cooperative multi-agent reinforcement-learning framework that combines causal influence with communication-driven consensus to encourage diligent agent behavior. This repository includes LM4C and communication-based baselines, with training configurations for MPE, Hallway, and SMACv2.

## Quick start

Run the following commands from the repository root:

```bash
conda create -n lm4c python=3.10 -y
conda activate lm4c
pip install -r requirements.txt
```

For GPU training, use a CUDA-enabled PyTorch installation compatible with your GPU driver. For SMACv2 training, also install StarCraft II and the required SMACv2 maps, then set `SC2PATH` to your StarCraft II installation directory:

```bash
export SC2PATH=/path/to/StarCraftII
```

## Training

The entry point is `src/main.py`. It combines `src/config/default.yaml`, the environment YAML, and the algorithm YAML in that order. Command-line values override the merged configuration. Specify configuration names relative to `src/config/algs/` and `src/config/envs/`, without the `.yaml` extension.

### LM4C

```bash
# MPE: Cooperative Navigation (CN)
python src/main.py \
  --alg-config=lm4c \
  --env-config=mpe/cooperation_navigation \
  --manual_seed=123 --cuda_id=0

# MPE: Predator-Prey (PP)
python src/main.py \
  --alg-config=lm4c \
  --env-config=mpe/prey_predator \
  --manual_seed=123 --cuda_id=0

# Hallway
python src/main.py \
  --alg-config=lm4c \
  --env-config=hallway/hallway \
  --manual_seed=123 --cuda_id=0

# SMACv2: Zerg 5 vs. 5
python src/main.py \
  --alg-config=lm4c \
  --env-config=smac_v2/smacv2_configs/zerg_5_vs_5 \
  --manual_seed=123 --cuda_id=0
```

Additional SMACv2 configurations cover Zerg, Protoss, and Terran teams at 5 vs. 5, 10 vs. 10, and 20 vs. 20. Select the corresponding YAML name, such as `smac_v2/smacv2_configs/protoss_10_vs_10`.

Use `--manual_seed` to set the random seed and `--cuda_id` to select the GPU. For a new training run, leave `checkpoint_path` empty and keep `evaluate=False`.

### Baselines

Baselines use the same entry point. For example, train TGCNet on CN with:

```bash
python src/main.py \
  --alg-config=tgcnet \
  --env-config=mpe/cooperation_navigation \
  --manual_seed=123 --cuda_id=0
```

To train another included baseline, replace `tgcnet` with `cola`, `sr`, `t2mac`, `masia`, or `laies`. Choose the desired environment with `--env-config`.

### Training configuration

LM4C's algorithm settings are defined in `src/config/algs/lm4c.yaml`; environment-specific settings and loss weights are defined in the selected environment YAML. Override individual values using Sacred's `with key=value` syntax:

```bash
python src/main.py \
  --alg-config=lm4c \
  --env-config=mpe/prey_predator \
  --manual_seed=123 --cuda_id=0 \
  with comm_rounds=3 t_max=10050000 batch_size_run=8
```

Common settings include `comm_rounds` (communication rounds), `t_max` (training timesteps), `batch_size_run` (parallel environments), `batch_size` (training batch size), and `lr` (learning rate). LM4C uses three communication rounds by default. To train on CPU, append `with use_cuda=False` to a training command.

### Models and logs

When `save_model=True`, checkpoints are saved under `results/models/<local_results_path>/<run_name>/<timestep>/`. The `save_model_interval` setting controls how often models are saved. LM4C run names follow `lm4c_seed<seed>_<timestamp>`.

When `use_tensorboard=True`, training logs are written under `results/tb_logs/<local_results_path>/<run_name>/`. Sacred stores experiment configurations and metrics under `results/sacred/`.

The result subdirectory comes from the environment configuration: PP uses `prey_predator`, Hallway uses `hallway_3a`, and the SMACv2 example uses `zerg_5_vs_5`. CN currently uses `cooperation_nevigation`, matching the `map_name` spelling in its YAML.

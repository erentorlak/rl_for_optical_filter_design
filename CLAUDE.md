# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is OML-PPO, a deep reinforcement learning system for automated multilayer optical design using Proximal Policy Optimization (PPO). The project uses PyTorch to train agents that design optical multilayer structures (like perfect absorbers, bandpass filters) by selecting materials and thicknesses to achieve target spectral properties.


Key dependencies managed by setup.py:
- `RLMultilayer` package requires: gym, tmm
- `gym_tmm0` package requires: gym, tmm

## Core Commands

### Training Experiments

Main training entry point with configurable options:
```bash
# Perfect absorber (6 layers max)
python ppo_absorber_visnir.py --cpu 1 --maxlen 6 --exp_name absorber6 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v0

# Perfect absorber (15 layers max)  
python ppo_absorber_visnir.py --cpu 1 --maxlen 15 --exp_name absorber15 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v1

# Bandpass filter
python ppo_absorber_visnir.py --env BandpassFilter-v0 --exp_name bandpass_test --maxlen 10



## Architecture Overview

### Key Components

1. **Training Pipeline**: `ppo_absorber_visnir.py` is the main entry point that configures and launches PPO training through SpinningUp's ExperimentGrid system.

2. **RL Environment**: `rl-multilayer/RLMultilayer/taskenvs/` contains:
   - `tasks.py`: Defines optical design tasks (perfect absorber, bandpass filter) with target spectra, materials, and wavelength ranges
   - `task_envs.py`: Wraps TMM simulation into Gym-compatible environment

3. **PPO Implementation**: `rl-multilayer/RLMultilayer/algos/ppo/` contains:
   - `ppo.py`: Core PPO training loop with rollouts, advantage calculation, and policy updates
   - `core.py`: Neural network architectures (MLP and RNN-based actor-critic models)

4. **Simulation Engine**: `rl-multilayer/RLMultilayer/utils.py` provides:
   - `TMM_sim`: Transfer Matrix Method simulation for multilayer optics
   - `cal_reward`: Reward function based on spectral target matching
   - Material database interface

5. **Material Data**: `data/` directory contains refractive index data for various materials (metals, dielectrics) used in optical designs.

### Experiment Structure

- **Experiments/**: Auto-generated directory containing logs, model checkpoints (`model.pt`), design trackers (`design_tracker_*.pkl`), and progress files
- Each experiment creates subdirectories based on configuration (e.g., `absorber6_max6_s42/`)
- Results include trained policies, design trajectories, and performance metrics

### Key Configuration Parameters

- `--maxlen`: Maximum number of layers in the optical stack
- `--use_rnn`: Use RNN-based actor-critic (vs MLP)
- `--discrete_thick`: Use discrete thickness values (vs continuous)
- `--hierarchical`: First select material type, then thickness
- `--spectrum_repr`: Use spectral representation in state space
- `--not_repeat`: Prevent material repetition in designs

### Custom Task Definition

To add new optical design tasks:
1. Add task function to `rl-multilayer/RLMultilayer/taskenvs/tasks.py`
2. Define target spectra, wavelength range, and available materials
3. Register environment name in `get_env_fn()`
4. Use `--env YourTaskName-v0` when training

## Dependencies


The project includes local copies of SpinningUp (`libs/spinningup/`) to avoid external dependencies.
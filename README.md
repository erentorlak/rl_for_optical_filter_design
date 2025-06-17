# OML-PPO: Optical Multilayer Design via PPO

This repository contains a **PyTorch implementation** of OML-PPO, a deep reinforcement learning method for automated multi-layer optical design, based on the paper:

> Wang, Haozhu, et al. "Automated multi-layer optical design via deep reinforcement learning." *Machine Learning: Science and Technology* 2.2 (2021): 025013. [[IOP Link](https://iopscience.iop.org/article/10.1088/2632-2153/abc327)]

---

## 📁 File Descriptions (with Purpose)

- **ppo_absorber_visnir.py**: Main entry point for training. Defines the command-line interface, parses arguments, sets up experiment configurations (like layer limits and environment parameters), and launches PPO training through the `ExperimentGrid` system.
- **rl-multilayer/RLMultilayer/**: Contains all components for defining custom reinforcement learning environments and models.
  - **algos/ppo/ppo.py**: The core PPO training loop — performs rollouts, calculates advantages, updates the policy and value networks, and manages logging and saving.
  - **algos/ppo/core.py**: Defines the architecture of policy networks (e.g., MLP and RNN-based actor-critic models) used in PPO.
  - **taskenvs/task_envs.py**: Wraps simulation logic to create a Gym-like environment that can be used for RL training.
  - **taskenvs/tasks.py**: Maps environment names (e.g., `PerfectAbsorberVisNIR-v0`) to simulation configurations — defines wavelengths, materials, layer thicknesses, and target absorption profiles.
  - **utils.py**: Provides utilities such as `TMM_sim` (for simulating light through layered structures), `cal_reward` (reward function), and various helper functions for spectra and merit calculation.
  - **__init__.py**: Declares this as a Python module.
- **libs/spinningup/**: A **local copy** of OpenAI's SpinningUp library. No need to clone it again. You can install it by running `pip install -e ./libs/spinningup` directly. This contains:
  - `run_utils.py`, `logx.py`, `mpi_tools.py`: experiment utilities and logging
  - PPO and other RL algorithms (some unused here)
- **Experiments/**: This is where logs, model checkpoints, `progress.txt` files, and trained policy data are saved automatically when training is launched.
- **setup.py**: Allows the `RLMultilayer/` module to be installed as a Python package via `pip install -e .`. This is required so that `from RLMultilayer...` imports work globally.
- **.gitignore**: Standard file to prevent unnecessary files (like logs or cache folders) from being committed to version control.
- **__pycache__/**: Python’s auto-generated cache files. These can safely be ignored or deleted at any time.

---

## 🧩 Features

- Custom Gym-compatible environments for multilayer optical design
- PPO implementation supporting MLP and RNN actor-critic architectures
- Configurable spectral representation, reward shaping, and hierarchical design
- Modular code with extensible architecture for custom optical tasks

---

## 📦 Requirements

> Tested with Python 3.7.4 and CPU-based training

### Required Python Packages
```txt
python==3.7.4
torch==1.3.1 or 1.4.0
torchvision==0.5.0
tmm==0.1.7
torchsummary
spinningup==0.2.0
mpi4py
matplotlib
seaborn
gym==0.15.7
tensorflow==1.15.5
protobuf==3.20.*
```

### 🔧 Installation Steps

```bash
# Clone the repo and navigate to the project root
cd rl-optical-design/

# Create and activate conda environment
conda create -n omlppo python=3.7.4
conda activate omlppo

# Install compatible versions of PyTorch & torchvision
pip install torch==1.4.0 torchvision==0.5.0

# Install core dependencies
pip install tmm==0.1.7 torchsummary matplotlib seaborn
conda install -c conda-forge mpi4py

# Fix TensorFlow protobuf compatibility
pip install protobuf==3.20.*
pip install tensorflow==1.15.5 gym==0.15.7

# Install the bundled SpinningUp version (no need to clone!)
pip install -e ./libs/spinningup

pip install -e ./rl-multilayer

pip install -e ./gym-tmm0  

  
```

---

## 🚀 Running Experiments

### Perfect Absorber (max 6 layers)
```bash

python ppo_1500nm_absorber.py --cpu 1 --maxlen 5 --exp_name erent_v1_absorber --use_rnn --discrete_thick

python ppo_absorber_visnir.py --cpu 1 --maxlen 6 --exp_name absorber7 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v0
  --cpu 4 \
  --maxlen 6 \
  --exp_name absorber6 \
  --use_rnn \
  --discrete_thick \
  --num_runs 1 \
  --env PerfectAbsorberVisNIR-v0
```

### Perfect Absorber (max 15 layers)
```bash
python ppo_absorber_visnir.py \
  --cpu 4 \
  --maxlen 15 \
  --exp_name absorber15 \
  --use_rnn \
  --discrete_thick \
  --num_runs 1 \
  --env PerfectAbsorberVisNIR-v1
```

### Custom Optical Task
To define your own optical design task (e.g. bandpass filter, anti-reflection coating):
1. Add a new function to `RLMultilayer/taskenvs/tasks.py`
2. Return a new environment config
3. Register its name in `get_env_fn()`
4. Pass that name via `--env` when launching

---


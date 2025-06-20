"""
Configurable Training Script for Selective Absorber Design
========================================================

This script provides an easy-to-configure interface for training PPO agents
to design selective optical absorbers at different wavelengths.

Simply modify the CONFIG section below to change parameters.
No terminal arguments needed - everything is controlled from code.
"""

from spinup.utils.run_utils import ExperimentGrid
from RLMultilayer.algos.ppo.ppo import ppo
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.tasks import get_env_fn
from RLMultilayer.utils import cal_reward, cal_reward_selective_1500, cal_reward_selective_configurable, cal_reward_selective_adaptive
import torch
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================
# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

CONFIG = {
    # --------------------------------------------------------------------------
    # Part 1: Defining the Optical Goal
    # These parameters set the specific engineering target for the AI.
    # --------------------------------------------------------------------------
    'peak_wavelength': 2000,
    # Purpose: The center wavelength (in nm) for the desired absorption peak.
    # Choices: Any wavelength within your material data range (e.g., 850, 1550, 2000).
    # Strategy: This is the primary knob for setting your design target.

    'absorption_window_width': 50,
    # Purpose: The total width (in nm) of the band where absorption should be maximized.
    # Choices: 20 (very sharp), 50 (sharp), 100 (moderate), 200 (broad).
    # Strategy: Use a narrow width for highly selective filters. A wider width
    # makes the problem easier but results in a less selective device.

    # --------------------------------------------------------------------------
    # Part 2: Training Duration and Complexity
    # These parameters control how long the AI trains and how complex the
    # generated structures can be.
    # --------------------------------------------------------------------------
    'epochs': 10,
    # Purpose: The total number of training cycles. Each epoch involves gathering
    # data and updating the AI's policy.
    # Choices: 10-50 (for quick tests), 200-1000 (for good results), 2000+ (for publication-quality).
    # Strategy: More epochs lead to better convergence but take longer. Start
    # with a moderate number and increase if the performance is still improving.

    'steps_per_epoch': 200,
    # Purpose: The number of simulation steps to run before updating the policy.
    # Choices: 500, 1000, 2000, 4000.
    # Strategy: Larger values provide more data for each update, leading to more
    # stable learning. 1000-2000 is a good balance of stability and speed.

    'num_runs': 1,
    # Purpose: The number of independent training runs to perform with different
    # random seeds.
    # Choices: 1 (for a single result), 3-5 (for statistical significance).
    # Strategy: Use 1 for initial exploration. Use 3+ for final experiments
    # to ensure your results are robust and not due to a single lucky run.

    'max_layers': 6,
    # Purpose: The maximum number of layers the AI is allowed to place.
    # Choices: 3-5 (simple), 6-12 (medium), 15+ (complex).
    # Strategy: This is a critical parameter. Start with a smaller number (6-8)
    # as it makes the search space manageable. Increase only if simpler
    # structures fail to meet your performance target.

    # --------------------------------------------------------------------------
    # Part 3: Environment and Model Architecture
    # These booleans control key architectural features of the model and environment.
    # The recommended settings from the paper are often the best choice.
    # --------------------------------------------------------------------------
    'discrete_thick': True,
    # Purpose: If True, the agent chooses from a predefined list of thicknesses.
    # If False, it would output a continuous value (not fully supported by this repo).
    # Strategy: Always keep this True for this codebase.

    'use_rnn': True,
    # Purpose: Use a Recurrent Neural Network (RNN) as the agent's brain.
    # Strategy: **Crucial for this problem.** An RNN has memory, allowing it to
    # make decisions based on the sequence of layers it has already placed.
    # Always set to True.

    'spectrum_repr': False,
    # Purpose: If True, the entire optical spectrum is fed back to the agent as
    # part of its observation.
    # Strategy: Keep this False. It dramatically increases the observation size
    # and training complexity, and is usually not necessary.

    'not_repeat': True, # Changed from False
    # Purpose: If True, prevents the agent from placing the same material
    # twice in a row (e.g., 'SiO2' followed by 'SiO2').
    # Strategy: **Highly Recommended to set to True.** This is a physically
    # sensible constraint that reduces the search space and speeds up learning.

    'hierarchical': True, # Changed from False
    # Purpose: If True, the agent makes a two-step decision: first pick a material,
    # then pick a thickness conditioned on that material.
    # Strategy: **Highly Recommended to set to True.** This mimics a more natural
    # design process and was shown to be effective in the paper.

    # --------------------------------------------------------------------------
    # Part 4: Reward Function (The "Teacher")
    # This determines how the agent is scored. It's the most important
    # part for guiding the AI to the correct solution.
    # Reward logic is now handled by the task itself.
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Part 5: Neural Network Hyperparameters
    # These are advanced settings for the AI's "brain". Usually safe to leave
    # at their default values.
    # --------------------------------------------------------------------------
    'hidden_sizes': (64,),
    # Purpose: The number and size of layers in the neural network.
    # Strategy: (64,) or (64, 64) is usually sufficient. Deeper networks
    # are not always better and can be harder to train.

    'cell_size': 64,
    # Purpose: The size of the RNN's memory cell.
    # Strategy: 64 or 128 is a standard choice.

    'act_emb': True,
    'act_emb_dim': 5,
    # Purpose: Related to the hierarchical model. Creates a learned "embedding"
    # for each material.
    # Strategy: Keep these defaults when using the hierarchical model.

    'channels': 16,
    # Purpose: Number of channels in the CNN layers of the neural network.
    # Strategy: 16 is the standard value used across all experiments in this codebase.

    # --------------------------------------------------------------------------
    # Part 6: PPO Algorithm Hyperparameters
    # These are core settings for the PPO learning algorithm. Best to leave
    # these at the values recommended by the paper unless you are an RL expert.
    # --------------------------------------------------------------------------
    'gamma': 1.0,
    # Purpose: Discount factor for future rewards.
    # Strategy: 1.0 is appropriate here because the episode ends when the
    # design is complete, and we care about the final quality, not speed.

    'beta': 0.01,
    # Purpose: The weight of the entropy bonus. Encourages exploration.
    # Strategy: If the agent gets stuck in a bad solution early, try increasing
    # this to 0.05. If it never converges, try decreasing to 0.005.

    'lam': 0.95,
    # Purpose: The lambda parameter for Generalized Advantage Estimation (GAE).
    # Strategy: Keep this value between 0.9 and 0.99. 0.95 is a robust default.

    'train_pi_iters': 5,
    # Purpose: Number of times to update the policy per epoch.
    # Strategy: 5-10 is a good range. More iterations can lead to better
    # policy improvement but slow down training.

    'pi_lr': 5e-5,
    # Purpose: The learning rate for the policy network.
    # Strategy: 5e-5 is a safe, standard value. If training is unstable,
    # try decreasing to 1e-5.

    'reward_factor': 1.0,
    # Purpose: A global multiplier for all rewards.
    # Strategy: Keep at 1.0 unless your reward values are extremely small
    # (e.g., < 0.01), in which case you might scale them up.

    # --------------------------------------------------------------------------
    # Part 7: Experiment Management
    # --------------------------------------------------------------------------
    'exp_name': 'lorentzian_absorber_2000nm_test',
    # Purpose: The base name for the results folder. Your script will append
    # the wavelength to this.
    # Strategy: Choose a descriptive name for your experiment.

    'cpu_cores': 1,
    # Purpose: Number of parallel environments to run.
    # Strategy: Set this to the number of physical CPU cores you have for
    # maximum data collection speed. If you have 8 cores, set this to 8.

    'save_experiments': True,
    # Purpose: Whether to save the results to the ./Experiments/ directory.
    # Strategy: Always keep this True so you don't lose your work.
}

# ============================================================================
# AUTOMATIC CONFIGURATION BASED ON PEAK WAVELENGTH
# ============================================================================

def get_task_name_and_config(peak_wavelength, absorption_window_width):
    """
    Determines the task name and any specific task configurations.
    For this script, it's fixed for Lorentzian Absorber.
    """
    task_name = 'LorentzianAbsorber2000nm-v0'
    task_config = {} # No extra config needed from here
    return task_name, task_config

def create_reward_function_config(reward_type, peak_wavelength, absorption_window_width,
                                 custom_config=None):
    """
    Returns an empty dictionary as the reward logic is handled by the task itself.
    """
    return {}

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run_training():
    """
    Run the training with the configured parameters.
    """

    print("=" * 80)
    print("LORENTZIAN ABSORBER TRAINING CONFIGURATION")
    print("=" * 80)

    # Generate dynamic configuration
    # peak_wavelength and absorption_window_width from CONFIG are now just for record-keeping if needed,
    # but not used by get_task_name_and_config for this script.
    task_name, task_config = get_task_name_and_config(
        CONFIG['peak_wavelength'],
        CONFIG['absorption_window_width']
    )

    # Create reward function and its configuration (will be empty)
    reward_config = create_reward_function_config(
        None, # reward_type is no longer used
        CONFIG['peak_wavelength'],
        CONFIG['absorption_window_width'],
        None # custom_config is no longer used
    )

    # Update experiment name with wavelength (already part of exp_name in CONFIG)
    exp_name = CONFIG['exp_name']

    # Print configuration summary
    print(f"Target Wavelength: {CONFIG['peak_wavelength']} nm")
    print(f"Absorption Window: ±{CONFIG['absorption_window_width']//2} nm around peak (nominal)")
    print(f"Task: {task_name}")
    # print(f"Reward Function: Defined by task {task_name}") # Reward function type removed from CONFIG
    print(f"Max Layers: {CONFIG['max_layers']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Steps per Epoch: {CONFIG['steps_per_epoch']}")
    print(f"Use RNN: {CONFIG['use_rnn']}")
    print(f"Discrete Thickness: {CONFIG['discrete_thick']}")
    print(f"Experiment Name: {exp_name}")
    print("=" * 80)

    # Environment configuration
    env_kwargs = {
        "discrete_thick": CONFIG['discrete_thick'],
        'spectrum_repr': CONFIG['spectrum_repr'],
        'bottom_up': False,
        # reward_config is empty, and task defines its own merit function.
        # **reward_config will add nothing here, which is correct.
    }

    # Set up experiment grid
    eg = ExperimentGrid(name=exp_name)
    eg.add('env_fn', get_env_fn(task_name, **env_kwargs)) # env_kwargs now correctly passed
    eg.add('seed', [42*(i+1) for i in range(CONFIG['num_runs'])])
    eg.add('epochs', CONFIG['epochs'])
    eg.add('steps_per_epoch', CONFIG['steps_per_epoch'])

    # Neural network architecture
    eg.add('ac_kwargs:hidden_sizes', [CONFIG['hidden_sizes']], 'hid')
    eg.add('ac_kwargs:cell_size', CONFIG['cell_size'], '')
    eg.add('ac_kwargs:not_repeat', [CONFIG['not_repeat']])
    eg.add('ac_kwargs:ortho_init', ['on'])
    eg.add('ac_kwargs:hierarchical', [CONFIG['hierarchical']])
    eg.add('ac_kwargs:channels', CONFIG['channels'])
    eg.add('ac_kwargs:act_emb', [CONFIG['act_emb']])
    eg.add('ac_kwargs:act_emb_dim', CONFIG['act_emb_dim'])
    eg.add('ac_kwargs:scalar_thick', [False])

    # PPO configuration
    eg.add('use_rnn', CONFIG['use_rnn'])
    eg.add('gamma', CONFIG['gamma'])
    eg.add('beta', [CONFIG['beta']])
    eg.add('lam', [CONFIG['lam']])
    eg.add('max_ep_len', [CONFIG['max_layers']], in_name=True)
    eg.add('actor_critic', core.RNNActorCritic if CONFIG['use_rnn'] else core.MLPActorCritic)
    eg.add("train_pi_iters", [CONFIG['train_pi_iters']])
    eg.add("pi_lr", [CONFIG['pi_lr']])
    eg.add('reward_factor', [CONFIG['reward_factor']])
    eg.add('spectrum_repr', [CONFIG['spectrum_repr']])

    # Prepare output directory
    if CONFIG['save_experiments']:
        output_dir = f'./Experiments/{exp_name}'
        os.makedirs(output_dir, exist_ok=True)

        # Save configuration for reference
        import json
        config_file = os.path.join(output_dir, 'training_config.json')
        # reward_config is empty, reflecting task-defined reward.
        # task_config is also empty for this specific setup.

        with open(config_file, 'w') as f:
            json.dump({
                'CONFIG': CONFIG,
                'task_name': task_name,
                'task_config_passed_to_get_env_fn': task_config, # This was empty
                'reward_config_from_script': reward_config, # This was empty
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Configuration saved to: {config_file}")

    print("\nStarting training...")
    print("=" * 80)

    # Run the experiment
    eg.run(ppo,
           num_cpu=CONFIG['cpu_cores'],
           data_dir=f'./Experiments/{exp_name}' if CONFIG['save_experiments'] else None,
           datestamp=False)

    print("=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    if CONFIG['save_experiments']:
        print(f"Results saved in: ./Experiments/{exp_name}/")
        print("Use eren_vis.py to visualize and evaluate the trained models.")

# ============================================================================
# QUICK CONFIGURATION PRESETS
# ============================================================================

def load_preset(preset_name):
    """
    Load predefined configuration presets for common scenarios.

    Args:
        preset_name: Name of the preset to load
    """

    presets = {
        'telecom_1550': {
            'peak_wavelength': 1550,
            'absorption_window_width': 100,
            'exp_name': 'telecom_absorber',
            'max_layers': 8,
            'epochs': 100,
            # 'reward_function': 'adaptive',  # No longer needed in CONFIG
        },

        'nir_980': {
            'peak_wavelength': 980,
            'absorption_window_width': 80,
            'exp_name': 'nir_980_absorber',
            'max_layers': 6,
            'epochs': 75,
            # 'reward_function': 'adaptive',
        },

        'visible_633': {
            'peak_wavelength': 633,
            'absorption_window_width': 60,
            'exp_name': 'visible_red_absorber',
            'max_layers': 4,
            'epochs': 60,
            # 'reward_function': 'adaptive',
        },

        'custom_configurable_example_removed': { # This preset concept is less relevant now
            # 'peak_wavelength': 1300,
            # 'absorption_window_width': 120,
            # 'exp_name': 'custom_absorber_example',
            # 'max_layers': 6,
            # 'epochs': 50,
            # 'reward_function': 'configurable', # This would require a different task or this script to pass reward info
            # 'custom_reward_config': { ... }
        },

        'quick_test_lorentzian': { # Example for this new script
            'peak_wavelength': 2000, # Matches task
            'absorption_window_width': 50, # Matches task's internal idea
            'epochs': 5,
            'steps_per_epoch': 100,
            'exp_name': 'quick_test_lorentzian',
            'max_layers': 3,
        }
    }

    if preset_name in presets:
        CONFIG.update(presets[preset_name])
        print(f"Loaded preset: {preset_name}")
        for key, value in presets[preset_name].items():
            print(f"  {key}: {value}")
    else:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(presets.keys())}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    # To use a preset, uncomment one of these lines:
    # load_preset('quick_test_lorentzian')

    # Or ensure CONFIG values at the top are set as desired.

    run_training()
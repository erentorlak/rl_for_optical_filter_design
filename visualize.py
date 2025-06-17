# %%


# %%
######################3

# %%
# Cell 1: Imports and Setup
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import json
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Essential Setup ---
# Define the main folder containing your experiment results
# Example: '/home/user/rl-optical-design/Experiments/exp_may1_v2_3000epoch'
# EXPERIMENT_FOLDER = '~/Desktop/different_projects/rl_optical_design/rl_opt_design1/rl-optical-design/Experiments/exp_may1_v2_3000epoch'
EXPERIMENT_FOLDER ='C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7' 
ENV_NAME = 'PerfectAbsorberVisNIR-v0'






# --- Optional: Add Project Root to Path ---
# If the notebook is not in the root directory of your project,
# you might need to add the project path to sys.path
# Adjust the path '../' if necessary
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '../')) # Adjust if your notebook is elsewhere
if PROJECT_ROOT not in sys.path:
    print(f"Adding {PROJECT_ROOT} to sys.path")
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------

# Import your project's modules AFTER potentially adding to path
try:
    from RLMultilayer.utils import (visualize_progress, summarize_res, combine_tracker,
                                   load_exp_res, DesignTracker, cal_reward, TMM_sim)
    from RLMultilayer.taskenvs import tasks
    import plotting  # Your plotting script
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the notebook is run from a location where RLMultilayer can be found,")
    print("or adjust the PROJECT_ROOT variable and uncomment the sys.path modification.")

# Expand user path (like ~)
RESULTS_DIR = os.path.expanduser(EXPERIMENT_FOLDER)

# Basic Plotting Style
plt.style.use('ggplot')
sns.set(style="darkgrid", font_scale=1.2)
matplotlib.rcParams['figure.figsize'] = (12, 6) # Default figure size

print(f"Analyzing results from: {RESULTS_DIR}")
print(f"Using environment: {ENV_NAME}")

# %%
# Cell 2: Load Experiment Data
# This uses the load_exp_res function from plotting.py which should:
# 1. Find all subdirectories corresponding to different runs/seeds.
# 2. Load 'progress.txt' from each.
# 3. Load 'config.json' to get hyperparameters.
# 4. Load 'design_tracker_*.pkl', combine them, and find the best design string.
# 5. Combine everything into a single Pandas DataFrame.

print(f"Loading data from subdirectories within: {RESULTS_DIR}")
try:
    df_results = plotting.load_exp_res(RESULTS_DIR)
    print(f"Successfully loaded data for {len(df_results['seed'].unique())} seeds.")
    print("\nDataFrame Info:")
    df_results.info()
    print("\nDataFrame Head:")
    display(df_results.head())
    print("\nUnique Hyperparameters found (example: ac_kwargs:hidden_sizes):")
    # Display columns that might represent hyperparameters
    config_cols = [col for col in df_results.columns if ':' in col or col in ['gamma', 'beta', 'lam', 'pi_lr', 'vf_lr', 'epochs', 'max_ep_len', 'use_rnn']]
    for col in config_cols:
         if col in df_results.columns:
             print(f"- {col}: {df_results[col].unique()}")
except FileNotFoundError:
    print(f"Error: Could not find results files (progress.txt, config.json, etc.) in {RESULTS_DIR} or its subdirectories.")
    print("Please ensure EXPERIMENT_FOLDER points to the correct location.")
    df_results = None
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    df_results = None

# %%
# Cell 3: Visualize Overall Training Trends (Across Seeds)
# Uses visualize_results from plotting.py

if df_results is not None:
    print("Plotting training trends across different seeds...")
    # This function plots MaxEpRet, AverageEpRet, Entropy, EpLen vs. Epoch
    # and returns the best design string found for each seed.
    best_designs_strings = plotting.visualize_results(RESULTS_DIR, x='Epoch')
    plt.tight_layout()
    plt.show()

    print("\nBest design strings found per seed:")
    for i, design_str in enumerate(best_designs_strings):
        print(f"Seed {i+1}: {design_str[0] if design_str else 'N/A'}")
else:
    print("Skipping training visualization because data loading failed.")
    best_designs_strings = []

# %%
# Cell 4: Identify and Display the Overall Best Design(s)

if df_results is not None and best_designs_strings:
    print("Extracting and ranking the best designs found...")
    try:
        # Use the extract_designs function from plotting.py
        m0s, x0s, merits = plotting.extract_designs(best_designs_strings)

        if merits:
            best_idx = np.argmax(merits)
            m0_best = m0s[best_idx]
            x0_best = x0s[best_idx]
            merit_best = merits[best_idx]

            print(f"\n--- Overall Best Design Found ---")
            print(f"  Materials: {m0_best}")
            print(f"  Thicknesses (nm): {x0_best}")
            print(f"  Merit (Reward): {merit_best:.5f}")
            print(f"  Source Seed Index: {best_idx}") # Note: This is index in the unique list
        else:
            print("Could not extract valid designs or merits.")
            m0_best, x0_best, merit_best = None, None, None

    except Exception as e:
        print(f"Error extracting designs: {e}")
        m0_best, x0_best, merit_best = None, None, None

else:
    print("Skipping best design identification.")
    m0_best, x0_best, merit_best = None, None, None

# %%
import importlib
import RLMultilayer.utils as utils
importlib.reload(utils)

# %%
# Cell 5: Simulate and Plot the Spectrum of the Best Design

# Need to recreate the environment to get the simulator
# --- Determine Env Kwargs (Try to load from config or use defaults) ---
env_kwargs = {'discrete_thick': True, 'spectrum_repr': False, 'bottom_up': False} # Sensible defaults
try:
    # Find one config.json to infer kwargs (assuming they are consistent)
    first_run_dir = next(d for d in glob.glob(os.path.join(RESULTS_DIR, '*')) if os.path.isdir(d))
    config_path = os.path.join(first_run_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Update kwargs based on config if keys exist
            if 'env_kwargs' in config:
                 env_kwargs.update(config['env_kwargs'])
            else: # Check top level args
                if 'discrete_thick' in config: env_kwargs['discrete_thick'] = config['discrete_thick']
                if 'spectrum_repr' in config: env_kwargs['spectrum_repr'] = config['spectrum_repr']
                if 'bottom_up' in config: env_kwargs['bottom_up'] = config['bottom_up']
            print(f"Inferred env_kwargs from {config_path}: {env_kwargs}")
    else:
        print("config.json not found, using default env_kwargs.")

except (StopIteration, FileNotFoundError):
     print("Could not find a config.json in subdirectories, using default env_kwargs.")
except Exception as e:
     print(f"Error reading config.json: {e}. Using default env_kwargs.")
# --------------------------------------------------------------------

print(f"\nCreating environment '{ENV_NAME}' with kwargs: {env_kwargs}")
try:
    env_fn = tasks.get_env_fn(ENV_NAME, **env_kwargs)
    env = env_fn()
    simulator = env.simulator
    target = env.target
    print("Environment and simulator created successfully.")

    if m0_best and x0_best:
        print("\nSimulating the best design...")
        plt.figure()
        # Note: TMM_sim expects thickness list *without* np.inf at ends
        # But requires it *with* np.inf for the simulation call
        thickness_with_inf = [np.inf] + x0_best + [np.inf]
        R, T, A = simulator.spectrum(m0_best, thickness_with_inf, plot=True, title=False)

        # Construct title manually
        title_str = ' | '.join([f'{m} {d}nm' for m, d in zip(m0_best, x0_best)])
        if simulator.substrate != 'Air':
             title_str = f'Air | {title_str} | {simulator.substrate_thick}nm {simulator.substrate} | Air'
        else:
             title_str = f'Air | {title_str} | Air'
        title_str += f'\n Merit: {merit_best:.4f}'
        plt.title(title_str, fontsize=10)
        plt.show()

        # Verify reward calculation
        calculated_reward = cal_reward(R, T, A, target)
        print(f"Calculated reward from simulation: {calculated_reward:.5f}")
        if not np.isclose(calculated_reward, merit_best, atol=1e-4):
            print(f"Warning: Calculated reward {calculated_reward:.5f} differs slightly from stored merit {merit_best:.5f}")

    else:
        print("Skipping simulation as no best design was identified.")

except Exception as e:
    print(f"Error during environment creation or simulation: {e}")
    env = None # Mark env as invalid

# %%
# Cell 6: (Optional) Fine-tune the Best Design using Optimization

# This uses SciPy's optimizer (likely L-BFGS-B via `minimize`)
# to tweak the thicknesses of the best design found by RL.

# --- Configuration ---
FINETUNE_ENABLED = True # Set to False to skip this cell
MAX_THICKNESS_BOUND = 200 # Upper bound for thickness during optimization (nm)
# -------------------

if FINETUNE_ENABLED and env and m0_best and x0_best:
    print("\n--- Fine-tuning the best design ---")
    try:
        # Define bounds: e.g., [(min_thick_1, max_thick_1), (min_thick_2, max_thick_2), ...]
        # Using a constant range [15, MAX_THICKNESS_BOUND] for all layers here
        bounds = [(15, MAX_THICKNESS_BOUND)] * len(x0_best)
        print(f"Initial thicknesses: {x0_best}")
        print(f"Using bounds: {bounds}")

        # Use the finetune function from plotting.py
        x_opt, res_opt = plotting.finetune(
            simulator=env.simulator,
            m0=m0_best,
            x0=x0_best,
            target=env.target,
            display=True, # This will plot before/after spectra
            bounds=bounds
        )

        optimized_merit = 1 - res_opt.fun
        print(f"\nOptimization Result: {'Success' if res_opt.success else 'Failed'}")
        print(f"  Optimized Thicknesses (nm): {x_opt}")
        print(f"  Optimized Merit (Reward): {optimized_merit:.5f}")
        print(f"  Improvement: {optimized_merit - merit_best:.5f}")

    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")

elif not FINETUNE_ENABLED:
    print("Fine-tuning is disabled.")
else:
    print("Skipping fine-tuning because environment/best design is not available.")

# %%
# Cell 7: (Optional) Batch Fine-tune All Unique Best Designs

# This is useful if visualize_results showed several different promising
# designs across different seeds. It applies finetune to all unique
# top designs found in the experiment runs.

# --- Configuration ---
BATCH_FINETUNE_ENABLED = True # Set to True to run this cell
MAX_THICKNESS_BOUND_BATCH = 200 # Upper bound for thickness during optimization (nm)
# -------------------

if BATCH_FINETUNE_ENABLED and env and df_results is not None:
    print("\n--- Batch Fine-tuning Unique Best Designs ---")
    # Note: This function requires the main DataFrame df_results
    try:
        x_opts_batch, merits_opt_batch = plotting.batch_finetune(
            df=df_results,
            env=env,
            max_thick=MAX_THICKNESS_BOUND_BATCH
        )
        print("\nBatch fine-tuning complete.")
        print(f"Number of unique designs fine-tuned: {len(x_opts_batch)}")
        if merits_opt_batch:
             print(f"Best merit achieved after batch fine-tuning: {np.max(merits_opt_batch):.5f}")

    except Exception as e:
        print(f"An error occurred during batch fine-tuning: {e}")

elif not BATCH_FINETUNE_ENABLED:
    print("Batch fine-tuning is disabled.")
else:
    print("Skipping batch fine-tuning because environment or results dataframe is not available.")

# %%
# Cell 8: (Optional) Compare Across Hyperparameters
# This is most useful if your EXPERIMENT_FOLDER contains results
# generated by ExperimentGrid, meaning subfolders represent different
# hyperparameter settings rather than just different seeds of the same setting.

# --- Configuration ---
COMPARE_HPARAMS_ENABLED = True # Set to True to run this cell

# List of hyperparameter column names (from df_results.columns) to compare
# Example: ['ac_kwargs:hidden_sizes', 'ac_kwargs:hierarchical', 'use_rnn']
HPARAMS_TO_COMPARE = ['max_ep_len', 'ac_kwargs:not_repeat']

# Short names/abbreviations for the legend corresponding to HPARAMS_TO_COMPARE
# Example: ['Hid', 'Hier', 'RNN']
HPARAM_ABBREVIATIONS = ['MaxL', 'NoRep']
# -------------------


if COMPARE_HPARAMS_ENABLED and df_results is not None:
    print(f"\n--- Comparing Results Across Hyperparameters: {HPARAMS_TO_COMPARE} ---")

    # Verify the columns exist
    valid_hparams = [hp for hp in HPARAMS_TO_COMPARE if hp in df_results.columns]
    valid_abbrs = [ab for hp, ab in zip(HPARAMS_TO_COMPARE, HPARAM_ABBREVIATIONS) if hp in df_results.columns]

    if len(valid_hparams) != len(HPARAMS_TO_COMPARE):
        print("Warning: Some specified hyperparameters not found in DataFrame columns.")
        print(f"Using valid columns: {valid_hparams}")

    if valid_hparams:
        try:
            # Use the compare_across_hparams function from plotting.py
            # This plots AverageEpRet, MaxEpRet, and Entropy, grouped by the hyperparameter combinations.
            # It also prints summary statistics for the last epochs.
            plotting.compare_across_hparams(
                folder=RESULTS_DIR, # Pass the main folder again
                hparams=valid_hparams,
                abbrs=valid_abbrs
            )
        except Exception as e:
            print(f"An error occurred during hyperparameter comparison: {e}")
    else:
        print("No valid hyperparameter columns found to compare.")

elif not COMPARE_HPARAMS_ENABLED:
    print("Hyperparameter comparison is disabled.")
else:
    print("Skipping hyperparameter comparison because data loading failed.")

# %%


# %%
# Cell 5b: Plot Angle-Dependent Absorption Map for the Best Design

# --- Configuration ---
ANGLE_MAP_ENABLED = True # Set to False to skip this cell
MIN_ANGLE = 0           # Minimum incidence angle (degrees)
MAX_ANGLE = 70          # Maximum incidence angle (degrees)
ANGLE_STEP = 1          # Step size for angles (degrees)
COLOR_MAP = 'jet'       # Colormap ('jet', 'viridis', 'plasma', 'inferno', etc.)
# -------------------

if ANGLE_MAP_ENABLED and env and m0_best and x0_best: # Check if env and best design exist
    print("\n--- Calculating Angle-Dependent Absorption ---")
    print(f"Angle range: {MIN_ANGLE} to {MAX_ANGLE} degrees, step: {ANGLE_STEP}")

    try:
        simulator = env.simulator # Get the simulator from the env object
        target = env.target       # Get the target

        # Prepare the structure details
        thickness_with_inf = [np.inf] + x0_best + [np.inf]
        wavelengths_nm = simulator.wavelength * 1000 # Wavelengths in nm

        # Define the angles to simulate
        theta_list = np.arange(MIN_ANGLE, MAX_ANGLE + ANGLE_STEP, ANGLE_STEP)

        # Calculate spectra for each angle
        all_absorptions = []
        print("Calculating spectra for each angle...")
        # Add tqdm for progress bar if many angles/wavelengths
        from tqdm.notebook import tqdm
        for theta in tqdm(theta_list):
            R, T, A = simulator.spectrum(m0_best, thickness_with_inf, theta=theta, plot=False)
            all_absorptions.append(A)

        # Convert list of 1D arrays to a 2D array (angles x wavelengths)
        absorption_map_raw = np.array(all_absorptions)

        # Transpose for plotting (wavelengths x angles)
        absorption_map = absorption_map_raw.T

        print("Calculation complete. Plotting heatmap...")

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figsize if needed

        # Use pcolormesh for the heatmap
        # X = angles, Y = wavelengths, C = absorption values
        im = ax.pcolormesh(
            theta_list,
            wavelengths_nm,
            absorption_map,
            cmap=COLOR_MAP,
            vmin=0,
            vmax=1,
            shading='gouraud' # Use 'gouraud' or 'auto' for smoother shading
        )

        # Add labels and title
        ax.set_xlabel("Incidence Angle (degree)")
        ax.set_ylabel("Wavelength (nm)")
        # Construct title
        title_str = ' | '.join([f'{m} {d}nm' for m, d in zip(m0_best, x0_best)])
        ax.set_title(f"Angle-Dependent Absorption\n{title_str}", fontsize=10)


        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Absorption')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during angle map generation: {e}")

elif not ANGLE_MAP_ENABLED:
    print("Angle map generation is disabled.")
else:
    print("Skipping angle map generation because environment or best design is not available.")

# %%


# %%




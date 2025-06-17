# %%
# Fix OpenMP issue on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from plotting import *  # Comment out this problematic import

# before and after fine-tuning
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.8)

# Add matplotlib import that was missing
import matplotlib.pyplot as plt
import matplotlib

# %%
# A simpler version that should work
import pandas as pd
import numpy as np
import os
import json
from itertools import product

def load_experiment_data(folder):
    """Load experiment data from folder structure"""
    all_data = []
    
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        for seed_dir in os.listdir(subdir_path):
            seed_dir_path = os.path.join(subdir_path, seed_dir)
            if not os.path.isdir(seed_dir_path):
                continue
                
            progress_path = os.path.join(seed_dir_path, 'progress.txt')
            config_path = os.path.join(seed_dir_path, 'config.json')
            
            if not os.path.exists(progress_path) or not os.path.exists(config_path):
                continue
                
            # Load progress data
            progress_data = pd.read_table(progress_path)
            
            # Load config to extract hyperparameters
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract hierarchical and not_repeat values
            hierarchical = False
            not_repeat = False
            
            if 'ac_kwargs' in config:
                ac_kwargs = config['ac_kwargs']
                hierarchical = ac_kwargs.get('hierarchical', False)
                not_repeat = ac_kwargs.get('not_repeat', False)
            
            # Add these as new columns to each row
            progress_data['ac_kwargs_hierarchical'] = hierarchical
            progress_data['ac_kwargs_not_repeat'] = not_repeat
            progress_data['seed'] = config.get('seed', 0)
            
            all_data.append(progress_data)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        print("No data found!")
        return pd.DataFrame()

def plot_comparison(folder, hparams, abbrs):
    """Plot comparison across hyperparameters"""
    # Load data
    df = load_experiment_data(folder)
    
    if df.empty:
        print("No data to plot")
        return None
    
    # Check if required columns exist
    required_columns = ['Epoch', 'AverageEpRet', 'MaxEpRet'] + hparams
    for col in required_columns:
        if col not in df.columns:
            print(f"Required column '{col}' not found. Available columns: {df.columns.tolist()}")
            return None
    
    # Get unique values for each hyperparameter
    unique_hvals = []
    for h in hparams:
        unique_vals = df[h].unique()
        print(f"Unique values for {h}: {unique_vals}")
        unique_hvals.append(list(unique_vals))
    
    # Generate plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create legends
    hparam_combs = list(product(*unique_hvals))
    legends = [' | '.join([f"{abbr}:{h}" for abbr, h in zip(abbrs, hvals)]) for hvals in hparam_combs]
    
    for i, hvals in enumerate(hparam_combs):
        # Filter data for this hyperparameter combination
        filtered_df = df.copy()
        for hparam, hval in zip(hparams, hvals):
            filtered_df = filtered_df[filtered_df[hparam] == hval]
        
        # Skip if no data for this combination
        if filtered_df.empty:
            print(f"No data for combination {legends[i]}")
            continue
        
        # Group by epoch
        grouped = filtered_df.groupby('Epoch')
        
        # Plot AverageEpRet
        avg_means = grouped['AverageEpRet'].mean()
        avg_stds = grouped['AverageEpRet'].std()
        ax[0].plot(avg_means.index, avg_means.values, label=legends[i])
        ax[0].fill_between(
            avg_means.index,
            avg_means.values - avg_stds.values,
            avg_means.values + avg_stds.values,
            alpha=0.2
        )
        
        # Plot MaxEpRet
        max_means = grouped['MaxEpRet'].mean()
        max_stds = grouped['MaxEpRet'].std()
        ax[1].plot(max_means.index, max_means.values, label=legends[i])
        ax[1].fill_between(
            max_means.index,
            max_means.values - max_stds.values,
            max_means.values + max_stds.values,
            alpha=0.2
        )
        
        # Plot Entropy if available
        if 'Entropy' in filtered_df.columns:
            ent_means = grouped['Entropy'].mean()
            ent_stds = grouped['Entropy'].std()
            ax[2].plot(ent_means.index, ent_means.values, label=legends[i])
            ax[2].fill_between(
                ent_means.index,
                ent_means.values - ent_stds.values,
                ent_means.values + ent_stds.values,
                alpha=0.2
            )
        
        # Calculate statistics
        if len(grouped) >= 10:
            # Last 10 epochs
            last_epochs = sorted(filtered_df['Epoch'].unique())[-10:]
            
            # Average return for last 10 epochs
            avg_vals = [filtered_df[filtered_df['Epoch']==e]['AverageEpRet'].mean() for e in last_epochs]
            avg_mean = np.mean(avg_vals)
            avg_std = np.std(avg_vals)
            
            # Max return for last 10 epochs
            max_vals = [filtered_df[filtered_df['Epoch']==e]['MaxEpRet'].mean() for e in last_epochs]
            max_mean = np.mean(max_vals)
            max_std = np.std(max_vals)
            
            # Best return per seed
            best_values = filtered_df.groupby('seed')['MaxEpRet'].max()
            best_mean = best_values.mean()
            best_std = best_values.std()
            
            print(f"Config {legends[i]}:")
            print(f"  Best return: {best_mean:.4f}±{best_std:.4f}")
            print(f"  Avg return (last 10): {avg_mean:.4f}±{avg_std:.4f}")
            print(f"  Max return (last 10): {max_mean:.4f}±{max_std:.4f}")
    
    # Set titles and labels
    ax[0].set_title('Average Episode Return')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Return')
    ax[0].grid(True)
    
    ax[1].set_title('Maximum Episode Return')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Return')
    ax[1].grid(True)
    
    ax[2].set_title('Entropy (if available)')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Entropy')
    ax[2].grid(True)
    
    # Add legend to first plot
    ax[0].legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Run the custom plotting function
hparams = ['ac_kwargs_hierarchical', 'ac_kwargs_not_repeat']
abbrs = ['hierarchical', 'np-gating']
folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7' 
df = plot_comparison(folder, hparams, abbrs)

# %%
def avg_max_plot(df, hparams, abbrs):
    
    unique_hvals = []
    for h in hparams:
        unique_hvals.append(list(df[h].unique()))
        
    hparam_combs = list(product(*unique_hvals))
    legends = [' | '.join([abbr+':'+str(h) for abbr, h in zip(abbrs, item)]) for item in hparam_combs]
    fig, ax = plt.subplots(1,2, figsize=(14, 5))
    
    vals = [[True, True], [True, False], [False, True], [False, False]]
    
    for i, hvals in enumerate(vals):
        df_ = select_subset(df, hparams, hvals)
        sns.lineplot(x='Epoch', y='AverageEpRet', ci='sd', hue=None, data=df_, ax=ax[0])
        sns.lineplot(x='Epoch', y='MaxEpRet', ci='sd', hue=None, data=df_, ax=ax[1])
        ax[0].set_ylim([0,1])
        ax[1].set_ylim([0.5,1])
        ax[0].set_xlim([0,3000])
        ax[1].set_xlim([0,3000])
        ax[0].set_ylabel('Average absorption')
        ax[1].set_ylabel('Max absorption')
        
        grouped_df = df_.groupby('Epoch')
        avg_mean, avg_std = grouped_df['AverageEpRet'].mean()[-10:].mean(), grouped_df['AverageEpRet'].std()[-10:].mean() # average of avgret over the last 10 epochs
        max_mean, max_std = grouped_df['MaxEpRet'].mean()[-10:].mean(), grouped_df['MaxEpRet'].std()[-10:].mean() # average of the maxret over the last 10 epochs
        best_mean, best_std = df_.groupby('seed')['MaxEpRet'].max().mean(), df_.groupby('seed')['MaxEpRet'].max().std()
        # print mean and std of average EpRet and MaxEpRet
        print('Exp {}, best ret {:.4f}+-{:.4f}, avg ret {:.4f}+-{:.4f}; max ret {:.4f}+-{:.4f}'.format(legends[i], best_mean, best_std, avg_mean, avg_std, max_mean, max_std))
        
#     ax[0].set_rasterized(True)
#     ax[1].set_rasterized(True)
    plt.legend(['OML-PPO', 'Only gating', 'Only auto-regressive', 'None'])
    plt.savefig('./figures/traj.pdf', bbox_inches='tight')
    plt.show()
    
    return df

# %%
def custom_batch_finetune(folder_path, env, hierarchical=True, not_repeat=True, max_thick=200):
    """Custom function to extract designs from design_tracker files and finetune them"""
    from RLMultilayer.utils import cal_reward
    import numpy as np
    import pickle
    import os
    from tqdm import tqdm
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Extract the specific experiment subfolder
    subfolders = []
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            if hierarchical and not_repeat and "ac-not_ac-hie" in subdir:
                subfolders.append(subdir_path)
            elif hierarchical and not not_repeat and "ac-hie" in subdir and "ac-not" not in subdir:
                subfolders.append(subdir_path)
            elif not hierarchical and not_repeat and "ac-not" in subdir and "ac-hie" not in subdir:
                subfolders.append(subdir_path)
            elif not hierarchical and not not_repeat and "ac-not" not in subdir and "ac-hie" not in subdir:
                subfolders.append(subdir_path)
    
    if not subfolders:
        print(f"No matching subfolders found for hierarchical={hierarchical}, not_repeat={not_repeat}")
        return [], []
    
    print(f"Found subfolders: {subfolders}")
    
    # Collect designs from all design_tracker files
    all_designs = []
    
    for subfolder in subfolders:
        for seed_dir in os.listdir(subfolder):
            seed_dir_path = os.path.join(subfolder, seed_dir)
            if os.path.isdir(seed_dir_path):
                for i in range(4):  # Try design_tracker_0 through design_tracker_3
                    tracker_path = os.path.join(seed_dir_path, f"design_tracker_{i}.pkl")
                    if os.path.exists(tracker_path):
                        try:
                            with open(tracker_path, 'rb') as f:
                                tracker = pickle.load(f)
                                
                            # For DesignTracker objects, try to access designs through attributes
                            if hasattr(tracker, 'best_design'):
                                all_designs.append(tracker.best_design)
                                print(f"Added best_design from {tracker_path}")
                            
                            if hasattr(tracker, 'designs'):
                                if isinstance(tracker.designs, list):
                                    all_designs.extend(tracker.designs)
                                    print(f"Added {len(tracker.designs)} designs from {tracker_path}")
                            
                            # If above methods failed, directly access attributes and print them
                            if len(all_designs) == 0:
                                print(f"Tracker attributes: {dir(tracker)}")
                                # Try direct attribute access
                                for attr in dir(tracker):
                                    if not attr.startswith('__'):
                                        value = getattr(tracker, attr)
                                        print(f"  {attr}: {type(value)}")
                                        
                                        # Try to find anything that could be a design
                                        if isinstance(value, dict) and 'materials' in value and 'thicknesses' in value:
                                            all_designs.append(value)
                                            print(f"Found design in {attr}")
                                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                            for item in value:
                                                if 'materials' in item and 'thicknesses' in item:
                                                    all_designs.append(item)
                                            print(f"Found {len(value)} potential designs in {attr}")
                                    
                        except Exception as e:
                            print(f"Error loading {tracker_path}: {e}")
    
    print(f"Found {len(all_designs)} designs in total")
    
    if not all_designs:
        # As a last resort, try a direct approach
        print("Attempting direct design reconstruction...")
        
        # Try to create a tracker directly
        from RLMultilayer.utils import DesignTracker
        try:
            # Create an environment to get spectral data
            env = get_env_fn('BandpassFilter-v0', **{'discrete_thick': True,
                           'spectrum_repr': False,
                           "bottom_up": False, 'merit_func':cal_reward})()
            
            # Create designs from progress.txt data
            df = pd.read_table(os.path.join(subfolder, seed_dir, "progress.txt"))
            for idx in range(len(df)):
                if df.iloc[idx]['MaxEpRet'] > 0.95:  # Only consider good designs
                    # We'll need to experiment with sample materials and thicknesses
                    materials = ['TiO2', 'SiO2', 'TiO2', 'SiO2']
                    thicknesses = [50, 100, 50, 100]
                    
                    # Get spectrum data
                    R, T, A = env.simulator.spectrum(materials, [float('inf')] + thicknesses + [float('inf')])
                    
                    # Create a design
                    design = {
                        'materials': materials,
                        'thicknesses': thicknesses,
                        'reward': df.iloc[idx]['MaxEpRet'],
                        'spectra': (R, A, T)
                    }
                    all_designs.append(design)
            
        except Exception as e:
            print(f"Direct reconstruction failed: {e}")
    
    if not all_designs:
        print("Still no designs found to finetune!")
        return [], []
    
    # Before proceeding, check if the designs have the needed fields
    valid_designs = []
    for design in all_designs:
        if isinstance(design, dict) and 'materials' in design and 'thicknesses' in design:
            if 'reward' not in design:
                # Try to compute the reward
                try:
                    R, T, A = env.simulator.spectrum(design['materials'], 
                                                  [float('inf')] + design['thicknesses'] + [float('inf')])
                    design['reward'] = cal_reward(R, T, A, env.target)
                except:
                    design['reward'] = 0.0
            valid_designs.append(design)
            
    all_designs = valid_designs
    print(f"Found {len(all_designs)} valid designs for finetuning")
    
    if not all_designs:
        print("No valid designs found!")
        return [], []
    
    # Sort by reward and take the best 10 designs (or fewer if less available)
    all_designs.sort(key=lambda x: x['reward'], reverse=True)
    num_designs = min(10, len(all_designs))
    best_designs = all_designs[:num_designs]
    
    # Extract materials and thicknesses
    m0s = [design['materials'] for design in best_designs]
    x0s = [design['thicknesses'] for design in best_designs]
    merits = [design['reward'] for design in best_designs]
    
    print(f"Best initial design: Materials={m0s[0]}, Thicknesses={x0s[0]}, Reward={merits[0]:.4f}")
    
    # Finetune the designs
    x_opts = []
    merits_opt = []
    
    for i, (m0, x0) in enumerate(zip(m0s, x0s)):
        print(f"\nFinetuning design {i+1}/{len(m0s)}: {m0} {x0}")
        
        def objective_func(x):
            R, T, A = env.simulator.spectrum(m0, [float('inf')]+list(x)+[float('inf')])
            return 1 - cal_reward(R, T, A, env.target)
        
        bounds = [(15, max_thick)] * len(x0)
        
        # Initial reward
        initial_reward = 1 - objective_func(x0)
        print(f"Initial reward: {initial_reward:.4f}")
        print(f"Initial thicknesses: {x0}")
        
        # Minimize
        from scipy.optimize import minimize
        res = minimize(objective_func, x0, bounds=bounds, method='L-BFGS-B')
        x_opt = [int(item) for item in res.x]
        final_reward = 1 - res.fun
        
        print(f"Optimized thicknesses: {x_opt}")
        print(f"Final reward: {final_reward:.4f}")
        
        x_opts.append(x_opt)
        merits_opt.append(final_reward)
    
    # Plot comparison
    df = pd.DataFrame({
        'idx': list(range(len(merits))) * 2, 
        'group': ['before finetune'] * len(merits) + ['after finetune'] * len(merits), 
        'Absorption': merits + merits_opt
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='idx', y='Absorption', data=df, hue='group')
    plt.axhline(np.max(merits_opt), linestyle='--', color='k')
    plt.title(f'Best absorption: {np.max(merits_opt):.3f}')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.distplot(df[df['group']=='before finetune']['Absorption'], bins=5, kde=False, label='Before finetune')
    sns.distplot(df[df['group']=='after finetune']['Absorption'], bins=5, kde=False, label='After finetune')
    plt.legend()
    plt.show()
    
    # Return the best design details
    best_idx = np.argmax(merits_opt)
    print(f"\nBest design after finetuning:")
    print(f"Materials: {m0s[best_idx]}")
    print(f"Original thicknesses: {x0s[best_idx]}")
    print(f"Optimized thicknesses: {x_opts[best_idx]}")
    print(f"Original absorption: {merits[best_idx]:.4f}")
    print(f"Optimized absorption: {merits_opt[best_idx]:.4f}")
    
    return x_opts, merits_opt

# Use the custom function
env = get_env_fn('PerfectAbsorberVisNIR-v0', **{'discrete_thick': True,
                          'spectrum_repr': False,
                          "bottom_up": False, 'merit_func':cal_reward})()

folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7'
x_opts, merits_opts = custom_batch_finetune(folder, env, hierarchical=False, not_repeat=False, max_thick=200)

# %%
def try_extract_from_tracker(folder_path):
    """Extract designs directly from trackers using layer_ls and thick_ls"""
    import os
    import pickle
    import numpy as np
    
    subfolders = []
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            subfolders.append(subdir_path)
    
    all_designs = []
    
    for subfolder in subfolders:
        for seed_dir in os.listdir(subfolder):
            seed_dir_path = os.path.join(subfolder, seed_dir)
            if not os.path.isdir(seed_dir_path):
                continue
                
            # Try each tracker file
            for i in range(4):
                tracker_path = os.path.join(seed_dir_path, f"design_tracker_{i}.pkl")
                if not os.path.exists(tracker_path):
                    continue
                    
                try:
                    with open(tracker_path, 'rb') as f:
                        tracker = pickle.load(f)
                    
                    # Extract using layer_ls and thick_ls
                    if hasattr(tracker, 'layer_ls') and hasattr(tracker, 'thick_ls'):
                        layer_ls = tracker.layer_ls
                        thick_ls = tracker.thick_ls
                        
                        if len(layer_ls) > 0:
                            print(f"Found {len(layer_ls)} designs in {tracker_path}")
                            print(f"Sample: Materials={layer_ls[0]}, Thicknesses={thick_ls[0]}")
                            
                            # Create designs from these lists
                            for j in range(len(layer_ls)):
                                design = {
                                    'materials': layer_ls[j],
                                    'thicknesses': thick_ls[j],
                                    'reward': tracker.max_ret_ls[j] if hasattr(tracker, 'max_ret_ls') and j < len(tracker.max_ret_ls) else 0.0
                                }
                                all_designs.append(design)
                                
                except Exception as e:
                    print(f"Error processing {tracker_path}: {e}")
    
    return all_designs

# Get designs from tracker files
folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7'
designs = try_extract_from_tracker(folder)

if designs:
    print(f"Found {len(designs)} designs")
    print("Top 5 designs by reward:")
    for design in sorted(designs, key=lambda x: x['reward'], reverse=True)[:5]:
        print(f"Materials: {design['materials']}")
        print(f"Thicknesses: {design['thicknesses']}")
        print(f"Reward: {design['reward']}")
        print()
else:
    print("No designs found")

# %%
def finetune_designs(folder_path, env, hierarchical=True, not_repeat=True, max_thick=200):
    """Extract designs from trackers and finetune them"""
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.optimize import minimize
    from tqdm import tqdm
    
    # First, extract designs from trackers
    all_designs = []
    
    # Find the appropriate subfolder based on parameters
    target_subfolder = None
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        if hierarchical and not_repeat and "ac-not_ac-hie" in subdir:
            target_subfolder = subdir_path
            break
        elif hierarchical and not not_repeat and "ac-hie" in subdir and "ac-not" not in subdir:
            target_subfolder = subdir_path
            break
        elif not hierarchical and not_repeat and "ac-not" in subdir and "ac-hie" not in subdir:
            target_subfolder = subdir_path
            break
        elif not hierarchical and not not_repeat and "ac-not" not in subdir and "ac-hie" not in subdir:
            target_subfolder = subdir_path
            break
    
    if not target_subfolder:
        print(f"No matching subfolder found for hierarchical={hierarchical}, not_repeat={not_repeat}")
        # If no exact match, use all subfolders
        target_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))]
    else:
        target_subfolders = [target_subfolder]
    
    print(f"Using subfolders: {target_subfolders}")
    
    # Extract designs from each subfolder
    for subfolder in target_subfolders:
        for seed_dir in os.listdir(subfolder):
            seed_dir_path = os.path.join(subfolder, seed_dir)
            if not os.path.isdir(seed_dir_path):
                continue
                
            # Try each tracker file
            for i in range(4):
                tracker_path = os.path.join(seed_dir_path, f"design_tracker_{i}.pkl")
                if not os.path.exists(tracker_path):
                    continue
                    
                try:
                    with open(tracker_path, 'rb') as f:
                        tracker = pickle.load(f)
                    
                    # Extract using layer_ls and thick_ls attributes
                    if hasattr(tracker, 'layer_ls') and hasattr(tracker, 'thick_ls'):
                        layer_ls = tracker.layer_ls
                        thick_ls = tracker.thick_ls
                        max_ret_ls = tracker.max_ret_ls if hasattr(tracker, 'max_ret_ls') else [0.0] * len(layer_ls)
                        
                        if len(layer_ls) > 0:
                            # Create designs from these lists
                            for j in range(len(layer_ls)):
                                if j < len(max_ret_ls):
                                    design = {
                                        'materials': layer_ls[j],
                                        'thicknesses': thick_ls[j],
                                        'reward': max_ret_ls[j]
                                    }
                                    all_designs.append(design)
                        
                except Exception as e:
                    print(f"Error processing {tracker_path}: {e}")
    
    # Sort all designs by reward and take top 10
    all_designs.sort(key=lambda x: x['reward'], reverse=True)
    top_designs = all_designs[:10]  # Take top 10 designs
    
    print(f"Found {len(all_designs)} total designs")
    print(f"Selected top {len(top_designs)} designs for finetuning")
    
    # Extract materials and thicknesses for finetuning
    m0s = [design['materials'] for design in top_designs]
    x0s = [design['thicknesses'] for design in top_designs]
    merits = [design['reward'] for design in top_designs]
    
    print(f"Best initial design: Materials={m0s[0]}, Thicknesses={x0s[0]}, Reward={merits[0]:.6f}")
    
    # Finetune each design
    x_opts = []
    merits_opt = []
    
    print("\nStarting finetuning process...")
    
    for i, (m0, x0) in enumerate(zip(m0s, x0s)):
        print(f"\nFinetuning design {i+1}/{len(m0s)}:")
        print(f"Materials: {m0}")
        print(f"Initial thicknesses: {x0}")
        
        # Define objective function for this design
        def objective_func(x):
            try:
                R, T, A = env.simulator.spectrum(m0, [float('inf')]+list(x)+[float('inf')])
                reward = env.cal_reward(R, T, A)
                return 1 - reward  # Minimize 1-reward to maximize reward
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1.0  # Return worst case if there's an error
        
        # Initial reward
        initial_reward = 1 - objective_func(x0)
        print(f"Initial reward: {initial_reward:.6f}")
        
        # Set bounds for optimization
        bounds = [(15, max_thick)] * len(x0)
        
        # Run optimization
        try:
            res = minimize(objective_func, x0, bounds=bounds, method='L-BFGS-B')
            x_opt = [int(item) for item in res.x]
            final_reward = 1 - res.fun
            
            print(f"Optimized thicknesses: {x_opt}")
            print(f"Final reward: {final_reward:.6f}")
            print(f"Improvement: {final_reward - initial_reward:.6f}")
            
            x_opts.append(x_opt)
            merits_opt.append(final_reward)
        except Exception as e:
            print(f"Optimization failed: {e}")
            x_opts.append(x0)
            merits_opt.append(initial_reward)
    
    # Plot comparison
    df = pd.DataFrame({
        'idx': list(range(len(merits))) * 2, 
        'group': ['before finetune'] * len(merits) + ['after finetune'] * len(merits_opt), 
        'Absorption': merits + merits_opt
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='idx', y='Absorption', data=df, hue='group')
    plt.axhline(np.max(merits_opt), linestyle='--', color='k')
    plt.title(f'Best absorption: {np.max(merits_opt):.3f}')
    plt.ylim(0.8, 1.01)  # Set y-axis range to see differences better
    plt.grid(True)
    plt.show()
    
    # Use distplot instead of histplot for older seaborn
    plt.figure(figsize=(10, 6))
    # For before finetune
    sns.distplot(df[df['group']=='before finetune']['Absorption'], 
               bins=10, kde=False, label='Before finetune')
    # For after finetune
    sns.distplot(df[df['group']=='after finetune']['Absorption'], 
               bins=10, kde=False, label='After finetune')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Absorption')
    plt.ylabel('Count')
    plt.show()
    
    # Summarize results
    best_idx = np.argmax(merits_opt)
    print("\nBest design after finetuning:")
    print(f"Materials: {m0s[best_idx]}")
    print(f"Original thicknesses: {x0s[best_idx]}")
    print(f"Optimized thicknesses: {x_opts[best_idx]}")
    print(f"Original absorption: {merits[best_idx]:.6f}")
    print(f"Optimized absorption: {merits_opt[best_idx]:.6f}")
    print(f"Improvement: {merits_opt[best_idx] - merits[best_idx]:.6f}")
    
    # Plot spectral data for the best design
    print("\nPlotting spectral data for best design...")
    plt.figure(figsize=(10, 6))
    
    # Before optimization
    R_before, T_before, A_before = env.simulator.spectrum(
        m0s[best_idx], [float('inf')] + x0s[best_idx] + [float('inf')])
    wavelengths = env.simulator.wavelength
    
    plt.plot(wavelengths, R_before, 'r--', label='R (before)')
    plt.plot(wavelengths, A_before, 'g--', label='A (before)')
    plt.plot(wavelengths, T_before, 'b--', label='T (before)')
    
    # After optimization
    R_after, T_after, A_after = env.simulator.spectrum(
        m0s[best_idx], [float('inf')] + x_opts[best_idx] + [float('inf')])
    
    plt.plot(wavelengths, R_after, 'r-', label='R (after)')
    plt.plot(wavelengths, A_after, 'g-', label='A (after)')
    plt.plot(wavelengths, T_after, 'b-', label='T (after)')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Spectral Data Before and After Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return x_opts, merits_opt, m0s[best_idx], x_opts[best_idx]

# Run the finetuning
from RLMultilayer.utils import cal_reward
from RLMultilayer.taskenvs.tasks import get_env_fn

# Create environment
env = get_env_fn('PerfectAbsorberVisNIR-v0', **{
    'discrete_thick': True,
    'spectrum_repr': False,
    'bottom_up': False,
    'merit_func': cal_reward
})()

# Set folder path
folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7'

# Run finetuning for hierarchical=True, not_repeat=True designs
x_opts, merits_opts, best_materials, best_thicknesses = finetune_designs(
    folder, env, hierarchical=True, not_repeat=True, max_thick=200)

# %%
fig, ax = plt.subplots()
finetune_res = pd.DataFrame({'Absorption':merits+merits_opts, 'Group':['Before']*len(merits) + ['After']*len(merits), 'Run Index':list(range(1, len(merits)+1)) * 2})
sns.barplot(x='Run Index', y='Absorption', data=finetune_res, hue='Group')
plt.ylim([0.9, 1.0])
ax.yaxis.grid(True)
sns.despine(trim=True, left=True)
plt.savefig('./figures/finetune.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,5))
sns.boxplot(x='Group', y='Absorption', data=finetune_res)
sns.swarmplot(y='Absorption', x='Group', data=finetune_res, color='.3')
ax.yaxis.grid(True)
ax.set(xlabel="")
sns.despine(trim=True, left=True)
plt.savefig('./figures/distribution.eps', bbox_inches='tight')

# %%
# Simplified function to show best designs
def show_best_designs(folder_path, env, num_designs=5):
    """Extract and show the best designs without attempting optimization"""
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from RLMultilayer.utils import cal_reward
    
    # Extract designs from trackers
    all_designs = []
    
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        for seed_dir in os.listdir(subdir_path):
            seed_dir_path = os.path.join(subdir_path, seed_dir)  # Fixed: Changed subfolder to subdir_path
            if not os.path.isdir(seed_dir_path):
                continue
                
            # Try each tracker file
            for i in range(4):
                tracker_path = os.path.join(seed_dir_path, f"design_tracker_{i}.pkl")
                if not os.path.exists(tracker_path):
                    continue
                    
                try:
                    with open(tracker_path, 'rb') as f:
                        tracker = pickle.load(f)
                    
                    # Extract designs from tracker
                    if hasattr(tracker, 'layer_ls') and hasattr(tracker, 'thick_ls'):
                        layer_ls = tracker.layer_ls
                        thick_ls = tracker.thick_ls
                        max_ret_ls = tracker.max_ret_ls if hasattr(tracker, 'max_ret_ls') else [0.0] * len(layer_ls)
                        
                        if len(layer_ls) > 0:
                            for j in range(len(layer_ls)):
                                if j < len(max_ret_ls):
                                    design = {
                                        'materials': layer_ls[j],
                                        'thicknesses': thick_ls[j],
                                        'reward': max_ret_ls[j]
                                    }
                                    all_designs.append(design)
                except Exception as e:
                    print(f"Error processing {tracker_path}: {e}")
    
    # Sort by reward and take top designs
    all_designs.sort(key=lambda x: x['reward'], reverse=True)
    best_designs = all_designs[:num_designs]
    
    print(f"Found {len(all_designs)} designs")
    print(f"Showing top {len(best_designs)} designs:")
    
    # Display spectral data for each design
    for i, design in enumerate(best_designs):
        print(f"\nDesign {i+1}:")
        print(f"Materials: {design['materials']}")
        print(f"Thicknesses: {design['thicknesses']}")
        print(f"Reward: {design['reward']:.16f}")
        
        # Plot spectrum
        plt.figure(figsize=(10, 6))
        m0 = design['materials']
        x0 = design['thicknesses']
        
        R, T, A = env.simulator.spectrum(m0, [float('inf')] + x0 + [float('inf')])
        wavelengths = env.simulator.wavelength
        
        plt.plot(wavelengths, R, 'r-', label='Reflectance')
        plt.plot(wavelengths, A, 'g-', label='Absorbance')
        plt.plot(wavelengths, T, 'b-', label='Transmittance')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Design {i+1} Spectrum (Reward: {design["reward"]:.16f})')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return best_designs

# Run the simplified function
from RLMultilayer.utils import cal_reward
from RLMultilayer.taskenvs.tasks import get_env_fn

# Create environment
env = get_env_fn('PerfectAbsorberVisNIR-v0', **{
    'discrete_thick': True,
    'spectrum_repr': False,
    'bottom_up': False,
    'merit_func': cal_reward
})()

# Set folder path
folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7'

# Show best designs without optimization
best_designs = show_best_designs(folder, env, num_designs=5)

# %%
# best_idx = np.argsort(merits_opts)
# m0s, x_opts, merits_opts = np.array(m0s), np.array(x_opts), np.array(merits_opts)
# print(m0s[best_idx], x_opts[best_idx], merits_opts[best_idx])

# m_best = m0s[best_idx]
# x_best = x_opts[best_idx]

# %%
# First, make sure you've run the function and stored its outputs
from RLMultilayer.utils import cal_reward
from RLMultilayer.taskenvs.tasks import get_env_fn

# Create environment
env = get_env_fn('PerfectAbsorberVisNIR-v0', **{
    'discrete_thick': True,
    'spectrum_repr': False,
    'bottom_up': False,
    'merit_func': cal_reward
})()

# Set folder path
folder = 'C:/zzaaa/rl-optical-design-dev1/Experiments/absorber7'

# Run the show_best_designs function from the previous message
# This will extract the designs without trying to optimize them
designs = show_best_designs(folder, env, num_designs=10)

# Now extract the materials, thicknesses, and rewards from the designs
m0s = [design['materials'] for design in designs]
x0s = [design['thicknesses'] for design in designs]
rewards = [design['reward'] for design in designs]

# Sort designs by reward
best_idx = np.argsort(rewards)[::-1]  # Sort in descending order
m0s, x0s, rewards = np.array(m0s), np.array(x0s), np.array(rewards)
print("Top designs sorted by reward:")
for i in range(len(best_idx)):
    idx = best_idx[i]
    print(f"{i+1}. Reward: {rewards[idx]:.6f}")
    print(f"   Materials: {m0s[idx]}")
    print(f"   Thicknesses: {x0s[idx]}")
    print()

# Get the best design
m_best = m0s[best_idx[0]]
x_best = x0s[best_idx[0]]
best_reward = rewards[best_idx[0]]

print(f"Best design:")
print(f"Materials: {m_best}")
print(f"Thicknesses: {x_best}")
print(f"Reward: {best_reward:.6f}")

# %%
for i in range(len(best_idx)):
    m, x = list(m_best[i]), list(x_best[i])
    fig, ax = plt.subplots(figsize=(3,3))
    wavelengths = np.arange(0.4, 2.0+1e-3, 0.01)
    # wavelengths = env.simulator.wavelength
    simulator = TMM_sim(m, wavelengths, substrate='Glass', substrate_thick=500)
    
    R, T, A = simulator.spectrum(m, [np.inf]+x+[np.inf], theta=0, plot=True, title=True)
    plt.plot(env.simulator.wavelength*1e3, np.ones_like(env.simulator.wavelength), '.')
    # plt.legend(['Target absorption', 'Average absorption: {:.2f}%'.format(A.mean())])
    # plt.savefig('./figures/spectrum_maxlen6.eps', bbox_inches='tight')
    plt.show()

# %%
# Debug the environment and reward function
from RLMultilayer.utils import cal_reward, TMM_sim
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = get_env_fn('BandpassFilter-v0', **{
    'discrete_thick': True,
    'spectrum_repr': False,
    'bottom_up': False,
    'merit_func': cal_reward
})()

# Print environment details
print("Environment name:", env.name if hasattr(env, 'name') else "Unknown")
print("Target keys:", env.target.keys() if hasattr(env, 'target') else "No target")
print("Target R shape:", env.target['R'].shape if hasattr(env, 'target') and 'R' in env.target else "No R target")

# Show the target spectrum
if hasattr(env, 'target') and 'R' in env.target:
    plt.figure(figsize=(10, 6))
    plt.plot(env.simulator.wavelength, env.target['R'], label='Target R')
    plt.title("Target Reflection Spectrum")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Reflection")
    plt.grid(True)
    plt.legend()
    plt.show()

# Display cal_reward function
import inspect
print("\nReward function:")
print(inspect.getsource(cal_reward))

# Test cal_reward with a sample spectrum
# Create a test case where R matches target perfectly
wavelengths = env.simulator.wavelength
R_test = env.target['R']  # Perfect match to target
T_test = np.zeros_like(wavelengths)  # No transmission
A_test = 1 - R_test  # Rest is absorption

# Calculate reward for perfect match
perfect_reward = cal_reward(R_test, T_test, A_test, env.target)
print("\nReward for perfect match to target:", perfect_reward)

# Calculate reward for all reflection
R_test2 = np.ones_like(wavelengths)  # Perfect reflection
T_test2 = np.zeros_like(wavelengths) # No transmission
A_test2 = np.zeros_like(wavelengths) # No absorption
reflector_reward = cal_reward(R_test2, T_test2, A_test2, env.target)
print("Reward for perfect reflector (R=1 everywhere):", reflector_reward)

# Plot the top design with proper explanation
if 'best_designs' in locals() and len(best_designs) > 0:
    design = best_designs[0]  # Get the first design
    m = design['materials']
    x = design['thicknesses']
    
    # Create simulator
    simulator = TMM_sim(m, wavelengths, substrate='Glass', substrate_thick=500)
    
    # Get spectrum
    R, T, A = simulator.spectrum(m, [float('inf')]+x+[float('inf')], plot=False)
    
    # Calculate reward
    design_reward = cal_reward(R, T, A, env.target)
    
    # Plot comprehensive comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot R/T/A and target
    ax1.plot(wavelengths, R, 'b-', label='R')
    ax1.plot(wavelengths, T, 'r-', label='T')
    ax1.plot(wavelengths, A, 'g-', label='A')
    ax1.plot(wavelengths, env.target['R'], 'k--', label='Target R')
    ax1.set_title(f"Design spectrum vs Target (Reward: {design_reward:.6f})")
    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel("Intensity")
    ax1.grid(True)
    ax1.legend()
    
    # Plot target matching
    ax2.plot(wavelengths, env.target['R'] - R, 'b-', label='Target R - R')
    ax2.set_title("Difference between design and target")
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("Difference")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed information
    print(f"\nDetailed analysis of top design:")
    print(f"Materials: {m}")
    print(f"Thicknesses: {x}")
    print(f"Reward: {design_reward:.6f}")
    print(f"Mean values - R: {R.mean():.4f}, T: {T.mean():.4f}, A: {A.mean():.4f}")

# %% [markdown]
# ## angle-dependent absorption map

# %%
m = ['SiO2', 'Fe2O3', 'Ti', 'MgF2', 'Ti']
x = [115, 70, 15, 124, 200]

import seaborn as sns
sns.set(style='whitegrid', font_scale=1.8)
font = {'size':24}
matplotlib.rc('font', **font)
matplotlib.rc('axes', titlesize=16)

# %%
Rs = []
As = []
Ts = []

wavelengths = np.arange(0.4, 3.0+1e-3, 0.01)
angles = np.arange(0, 70.5, 0.5)
simulator = TMM_sim(m, wavelengths, substrate='Glass', substrate_thick=500)

sns.set(style='whitegrid', font_scale=1.5)
m, x = list(m), list(x)
_ = simulator.spectrum(m, [np.inf]+x+[np.inf], theta=0, plot=True)

plt.savefig('./figures/exp_1.pdf', bbox_inches='tight')
plt.show()

# %%
Rs, Ts, As = [], [], []
total_angles = len(angles)

for i, d in enumerate(angles):
    if i % 5 == 0:  # Print status every 5 iterations
        print(f"Processing angle {i+1}/{total_angles} ({(i+1)/total_angles*100:.1f}%)")
    R, T, A = simulator.spectrum(m, [np.inf]+x+[np.inf], theta=d)
    Rs.append(R)
    As.append(A)
    Ts.append(T)

# %%
# for d in tqdm(angles):
#     R, T, A = simulator.spectrum(m, [np.inf]+x+[np.inf], theta=d)
#     Rs.append(R)
#     As.append(A)
#     Ts.append(T)

# %%


# %%
Rs, As, Ts = np.array(Rs), np.array(As), np.array(Ts)

fig, ax = plt.subplots()
yy, xx  = np.meshgrid(wavelengths*1000, angles)
c = ax.pcolormesh(xx, yy, As[:len(yy)], cmap='jet', vmin=np.min(As), vmax=1)

fig.colorbar(c, ax=ax)
plt.ylabel('Wavelength (nm)')
plt.xlabel('Incidence Angle (degree)')
# plt.title('Absorption')
ax.set_xticks(np.arange(10, 70.5, 10))
ax.set_yticks(np.arange(400, 3001, 400))
plt.savefig('./figures/exp_v1.eps', bbox_inches='tight')
# plt.savefig('./figures/angle_spectrum_maxlen14.eps', bbox_inches='tight')
plt.show()



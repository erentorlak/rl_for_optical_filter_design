import pandas as pd
import json
from tqdm.notebook import tqdm
from itertools import product

import os
import numpy as np
import pickle as pkl
from gym import spaces
from scipy.optimize import minimize
from tqdm import tnrange

import sys
from RLMultilayer.utils import visualize_progress, summarize_res, combine_tracker, load_exp_res, DesignTracker, cal_reward
from RLMultilayer.taskenvs.tasks import get_env_fn
import glob

from torch import nn
import torch
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from RLMultilayer.taskenvs import tasks
from RLMultilayer.utils import cal_reward
from RLMultilayer.utils import TMM_sim

import seaborn as sns
sns.set(font_scale=1)
import re 

def finetune(simulator, m0, x0, target, display=False, bounds=None):
    '''
    Finetune the structure using quasi-Newton's method.
    
    Args:
        m0: materials list given by the upstream RL
        x0: thicknesses given by the upstream RL
        display: if true, then plot the spectrum before and after the finetuning.
        
    Returns:
        x_opt: finetuned thickness list
    '''
    
    def objective_func(x):
        R, T, A = simulator.spectrum(m0, [np.inf]+list(x)+[np.inf])
        return 1-cal_reward(R, T, A, target)
    
    if bounds is None:
        bounds = [(5, 200)] * len(x0)

    print('Initial reward {}'.format(1-objective_func(x0)))
    res = minimize(objective_func, x0, bounds=bounds, options={'disp':True})
    x_opt = [int(item) for item in res.x]
    
    if display:
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x0+[np.inf], title=True, plot=True)
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x_opt+[np.inf], title=True, plot=True)
    
    return x_opt, res

def plot_results(folder, col=None, row=None, hue=None):
    
    df = load_exp_res(folder)
    sns.set(font_scale=1)
    
    reward_factor = df['reward_factor']
    df['LowEpRet'] = (df['AverageEpRet'] - 0.5 * df['StdEpRet']) / reward_factor
    df['HighEpRet'] = (df['AverageEpRet'] + 0.5 * df['StdEpRet']) / reward_factor
    df['NormalizedAverageEpRet'] = df['AverageEpRet']  / reward_factor
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, sharey=False)
    g = g.map(plt.plot, "NormalizedAverageEpRet").add_legend()
    g = g.map(plt.fill_between, "Epoch", "LowEpRet", "HighEpRet" , **{'alpha':0.5}).add_legend()
    g.set_ylabels('AverageEpRet')

    df['NormalizedMaxEpRet'] = df['MaxEpRet'] / reward_factor
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, sharey=False)
    g = g.map(plt.plot, "Epoch", 'NormalizedMaxEpRet').add_legend()

    plt.figure()
    sns.lineplot(y='AverageVVals', x='Epoch', data=df, ci='sd')

    plt.figure()
    sns.lineplot(y='Entropy', x='Epoch', data=df, ci='sd')

    print(df.iloc[df['MaxEpRet'].to_numpy().argmax()]['best_design'])

    return df

# def visualize_results(folder, x=None):
    
#     if x is None:
#         x = 'Epoch'
    
#     df = load_exp_res(folder)
#     df['seed'] = ['$%s$' %item for item in df['seed']]
    
#     fig, ax = plt.subplots(2,2, figsize=(10, 10))
#     ax = ax.ravel()
#     sns.lineplot(x=x, y='MaxEpRet', data=df, hue='seed', ci='sd', legend=None, ax=ax[0])
#     sns.lineplot(x=x, y='AverageEpRet', data=df, hue='seed', ci='sd', legend=None, ax=ax[1])
#     sns.lineplot(x=x, y='Entropy', data=df, hue='seed', ci='sd',  ax=ax[2])
#     sns.lineplot(x=x, y='EpLen', data=df, hue='seed', ci='sd',  ax=ax[3])

#     best_designs = []
#     for s in df['seed'].unique():
#         best_designs.append(df[df['seed']==s]['best_design'].iloc[0])
        
#     return best_designs



import matplotlib.pyplot as plt # Make sure matplotlib is imported

def visualize_results(folder, x=None):
    """
    Visualizes training trends using Matplotlib directly to ensure compatibility
    with older seaborn versions. Plots MaxEpRet, AverageEpRet, Entropy, EpLen
    vs. the specified x-axis variable (default 'Epoch').
    Returns the best design string found for each seed.
    """
    if x is None:
        x = 'Epoch'

    df = load_exp_res(folder)
    # Keep seed as is, don't convert to string format for iteration
    # df['seed'] = ['$%s$' % item for item in df['seed']]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10)) # Adjusted figsize slightly
    ax = ax.ravel()

    unique_seeds = df['seed'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_seeds))) # Get distinct colors

    metrics_to_plot = ['MaxEpRet', 'AverageEpRet', 'Entropy', 'EpLen']
    titles = ['Max Episode Return', 'Average Episode Return', 'Policy Entropy', 'Episode Length']

    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame columns. Skipping plot.")
            ax[i].set_title(f"{title}\n(Data Unavailable)")
            continue

        for j, seed in enumerate(unique_seeds):
            seed_df = df[df['seed'] == seed].sort_values(by=x) # Ensure data is sorted by x-axis
            if not seed_df.empty:
                 # Use label=seed if you want a legend, otherwise None
                ax[i].plot(seed_df[x], seed_df[metric], color=colors[j], alpha=0.8, label=seed if i == 0 else None) # Add legend only to first plot
            else:
                print(f"Warning: No data found for seed {seed} for metric {metric}.")

        ax[i].set_title(title)
        ax[i].set_xlabel(x)
        ax[i].set_ylabel(metric)
        ax[i].grid(True, linestyle='--', alpha=0.6) # Add grid

    # Add a single legend outside the plots if needed (using labels from the first plot)
    if len(unique_seeds) < 10: # Only add legend if not too many seeds
         fig.legend(title="Seed", loc='center right', bbox_to_anchor=(1.05, 0.5))


    # Extract best designs (same logic as before)
    best_designs = []
    for s in unique_seeds: # Iterate using original seed values
        # Find the row with the maximum MaxEpRet for this seed
        seed_df = df[df['seed'] == s]
        if not seed_df.empty:
            best_row_for_seed = seed_df.loc[seed_df['MaxEpRet'].idxmax()]
            best_designs.append(best_row_for_seed['best_design'])
        else:
            best_designs.append(None) # Append None if no data for seed

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for legend if shown

    return best_designs




def extract_designs(best_designs):
    m0s = []
    x0s = []
    merits = []
    for i in range(len(best_designs)):
        design = best_designs[i][0].split(',')[0].split('|')
        m0 = [item.split(' ')[0] for item in design]
        x0 = [item.split(' ')[1] for item in design]
        merit = best_designs[i][0].split(',')[1].split(' ')[2]
        x0 = [int(item) for item in x0]

        m0s.append(m0)
        x0s.append(x0)
        merits.append(float(merit))
    
    return m0s, x0s, merits

def batch_finetune(df, env, max_thick=200):
    m0s, x0s, merits = extract_designs(np.unique(df['best_design'].to_numpy()).tolist())
    
    x_opts = []
    merits_opt = []
    for m0, x0 in tqdm(zip(m0s, x0s)):
        x_opt, res = finetune(env.simulator, m0, x0, env.target, bounds=[[15, max_thick]]*len(x0))
        merits_opt.append(1 - res.fun)
        x_opts.append(x_opt)
        print(merits, 1-res.fun)

    df = pd.DataFrame({'idx':list(range(len(merits))) * 2, 'group':['before finetune'] * len(merits) + ['after finetune'] * len(merits), 'Absorption':merits+merits_opt})

    sns.barplot(x='idx', y='Absorption', data=df, hue='group')
    # plt.ylim(0.9, 1.0)
    plt.axhline(np.max(merits_opt), linestyle='--', color='k')
    plt.title('Best absorption: {:.3f}'.format(np.max(merits_opt)))
    plt.show()

    sns.distplot(df[df['group']=='before finetune']['Absorption'], bins=5, kde=False)
    sns.distplot(df[df['group']=='after finetune']['Absorption'], bins=5, kde=False)
    plt.legend(['Before finetune', 'After finetune'])
    
    return x_opts, merits_opt

def select_subset(df, hparams, hvals):
    df_ = df.copy()
    for hparam, hval in zip(hparams, hvals):
        df_ = df_[df_[hparam] == hval]
    return df_

def compare_across_hparams(folder, hparams, abbrs):
    
    df = load_exp_res(folder)
    unique_hvals = []
    for h in hparams:
        unique_hvals.append(list(df[h].unique()))
        
    hparam_combs = list(product(*unique_hvals))
    legends = [' | '.join([abbr+':'+str(h) for abbr, h in zip(abbrs, item)]) for item in hparam_combs]
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    
    for i, hvals in enumerate(list(product(*unique_hvals))):
        df_ = select_subset(df, hparams, hvals)
        
        # Replace lineplot with direct matplotlib plotting
        # For AverageEpRet
        grouped_avg = df_.groupby('Epoch')['AverageEpRet']
        avg_mean = grouped_avg.mean()
        avg_std = grouped_avg.std()
        ax[0].plot(avg_mean.index, avg_mean.values, label=legends[i])
        ax[0].fill_between(avg_mean.index, 
                           avg_mean.values - avg_std.values,
                           avg_mean.values + avg_std.values,
                           alpha=0.3)
        
        # For MaxEpRet
        grouped_max = df_.groupby('Epoch')['MaxEpRet']
        max_mean = grouped_max.mean()
        max_std = grouped_max.std()
        ax[1].plot(max_mean.index, max_mean.values, label=legends[i])
        ax[1].fill_between(max_mean.index, 
                           max_mean.values - max_std.values,
                           max_mean.values + max_std.values,
                           alpha=0.3)
        
        # For Entropy
        if 'Entropy' in df_.columns:
            grouped_ent = df_.groupby('Epoch')['Entropy']
            ent_mean = grouped_ent.mean()
            ent_std = grouped_ent.std()
            ax[2].plot(ent_mean.index, ent_mean.values, label=legends[i])
            ax[2].fill_between(ent_mean.index, 
                              ent_mean.values - ent_std.values,
                              ent_mean.values + ent_std.values,
                              alpha=0.3)
        
        # Calculate statistics for reporting
        grouped_df = df_.groupby('Epoch')
        last_10_epochs = max(0, len(grouped_df) - 10)
        
        if last_10_epochs > 0:
            avg_epochs = list(grouped_df.groups.keys())[-10:]
            max_epochs = list(grouped_df.groups.keys())[-10:]
            
            # Get means for last 10 epochs
            avg_values = [grouped_df.get_group(epoch)['AverageEpRet'].mean() for epoch in avg_epochs]
            max_values = [grouped_df.get_group(epoch)['MaxEpRet'].mean() for epoch in max_epochs]
            
            # Calculate statistics
            avg_mean_val = np.mean(avg_values)
            avg_std_val = np.std(avg_values)
            max_mean_val = np.mean(max_values)
            max_std_val = np.std(max_values)
            
            # Get best return across all seeds
            best_by_seed = df_.groupby('seed')['MaxEpRet'].max()
            best_mean = best_by_seed.mean()
            best_std = best_by_seed.std()
            
            # Print results
            print('Exp {}, best ret {:.4f}+-{:.4f}, avg ret {:.4f}+-{:.4f}; max ret {:.4f}+-{:.4f}'.format(
                legends[i], best_mean, best_std, avg_mean_val, avg_std_val, max_mean_val, max_std_val))
    
    # Set titles
    ax[0].set_title('Average Episode Return')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Return')
    
    ax[1].set_title('Maximum Episode Return')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Return')
    
    ax[2].set_title('Entropy')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Entropy')
    
    # Add legend to the rightmost plot
    ax[2].legend(legends)
    plt.tight_layout()
    plt.show()
    
    return df
    

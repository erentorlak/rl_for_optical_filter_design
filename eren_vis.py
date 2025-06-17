# %%
"""
Streamlined End-to-End RL Optical Design Analysis
=================================================
This script provides a complete analysis pipeline for RL optical design experiments.
"""

# Fix OpenMP issue on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import glob
from scipy.optimize import minimize
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, use a simple replacement
    def tqdm(iterable, desc="Progress"):
        print(f"{desc}...")
        return iterable
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set style
sns.set(style='whitegrid', font_scale=1.4)
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
EXPERIMENT_FOLDER = 'C:/zzaaa/rl-optical-design-dev1/Experiments/erent_v1_absorber'
ENV_NAME = 'erent_1500_absorber_task_v0'



class OpticalDesignAnalyzer:
    """Complete analysis pipeline for optical design experiments"""
    
    def __init__(self, experiment_folder, env_name):
        self.experiment_folder = experiment_folder
        self.env_name = env_name
        self.data = None
        self.env = None
        self.best_designs = []
        
    def load_data(self):
        """Load all experiment data from folder structure"""
        print(f"Loading data from: {self.experiment_folder}")
        
        if not os.path.exists(self.experiment_folder):
            print(f"Folder {self.experiment_folder} does not exist!")
            return False
            
        all_data = []
        all_designs = []
        
        for subdir in os.listdir(self.experiment_folder):
            subdir_path = os.path.join(self.experiment_folder, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            for seed_dir in os.listdir(subdir_path):
                seed_path = os.path.join(subdir_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                    
                # Load progress data
                progress_file = os.path.join(seed_path, 'progress.txt')
                config_file = os.path.join(seed_path, 'config.json')
                
                if os.path.exists(progress_file) and os.path.exists(config_file):
                    try:
                        # Load training progress
                        progress = pd.read_table(progress_file)
                        
                        # Load configuration
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        # Add metadata
                        progress['seed'] = config.get('seed', 0)
                        progress['max_ep_len'] = config.get('max_ep_len', 0)
                        
                        if 'ac_kwargs' in config:
                            ac_kwargs = config['ac_kwargs']
                            progress['hierarchical'] = ac_kwargs.get('hierarchical', False)
                            progress['not_repeat'] = ac_kwargs.get('not_repeat', False)
                        
                        all_data.append(progress)
                        
                        # Load design trackers
                        for i in range(4):
                            tracker_file = os.path.join(seed_path, f'design_tracker_{i}.pkl')
                            if os.path.exists(tracker_file):
                                try:
                                    with open(tracker_file, 'rb') as f:
                                        tracker = pickle.load(f)
                                    
                                    if hasattr(tracker, 'layer_ls') and hasattr(tracker, 'thick_ls'):
                                        layer_ls = tracker.layer_ls
                                        thick_ls = tracker.thick_ls
                                        reward_ls = getattr(tracker, 'max_ret_ls', [0] * len(layer_ls))
                                        
                                        for j in range(len(layer_ls)):
                                            if j < len(reward_ls):
                                                design = {
                                                    'materials': layer_ls[j],
                                                    'thicknesses': thick_ls[j],
                                                    'reward': reward_ls[j],
                                                    'seed': config.get('seed', 0)
                                                }
                                                all_designs.append(design)
                                except Exception as e:
                                    print(f"Error loading tracker {tracker_file}: {e}")
                    
                    except Exception as e:
                        print(f"Error loading data from {seed_path}: {e}")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            self.best_designs = sorted(all_designs, key=lambda x: x['reward'], reverse=True)[:10]
            print(f"Loaded {len(self.data)} training records and {len(self.best_designs)} best designs")
            return True
        else:
            print("No data found!")
            return False
    
    def setup_environment(self):
        """Setup the optical simulation environment"""
        try:
            # Add project to path if needed
            project_root = os.path.abspath('.')
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from RLMultilayer.taskenvs.tasks import get_env_fn
            from RLMultilayer.utils import cal_reward
            
            env_kwargs = {
                'discrete_thick': True,
                'spectrum_repr': False,
                'bottom_up': False,
                'merit_func': cal_reward
            }
            
            env_fn = get_env_fn(self.env_name, **env_kwargs)
            self.env = env_fn()
            print("Environment setup successful")
            return True
            
        except Exception as e:
            print(f"Environment setup failed: {e}")
            return False
    
    def plot_training_progress(self, save_to_file=False):
        """Plot training progress across all runs"""
        if self.data is None:
            print("No data loaded")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get unique seeds for different colors
        seeds = self.data['seed'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(seeds)))
        
        # Average Return
        for i, seed in enumerate(seeds):
            seed_data = self.data[self.data['seed'] == seed]
            axes[0,0].plot(seed_data['Epoch'], seed_data['AverageEpRet'], 
                          color=colors[i], alpha=0.7, linewidth=1.5)
        axes[0,0].set_title('Average Episode Return')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Average Return')
        axes[0,0].grid(True)
        
        # Max Return
        for i, seed in enumerate(seeds):
            seed_data = self.data[self.data['seed'] == seed]
            axes[0,1].plot(seed_data['Epoch'], seed_data['MaxEpRet'], 
                          color=colors[i], alpha=0.7, linewidth=1.5)
        axes[0,1].set_title('Maximum Episode Return')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Max Return')
        axes[0,1].grid(True)
        
        # Episode Length
        if 'EpLen' in self.data.columns:
            for i, seed in enumerate(seeds):
                seed_data = self.data[self.data['seed'] == seed]
                axes[1,0].plot(seed_data['Epoch'], seed_data['EpLen'], 
                              color=colors[i], alpha=0.7, linewidth=1.5)
            axes[1,0].set_title('Episode Length')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Episode Length')
            axes[1,0].grid(True)
        else:
            axes[1,0].text(0.5, 0.5, 'EpLen not available', 
                          transform=axes[1,0].transAxes, ha='center', va='center')
            axes[1,0].set_title('Episode Length (N/A)')
        
        # Entropy
        if 'Entropy' in self.data.columns:
            for i, seed in enumerate(seeds):
                seed_data = self.data[self.data['seed'] == seed]
                axes[1,1].plot(seed_data['Epoch'], seed_data['Entropy'], 
                              color=colors[i], alpha=0.7, linewidth=1.5)
            axes[1,1].set_title('Policy Entropy')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Entropy')
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Entropy not available', 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title('Policy Entropy (N/A)')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        final_epoch = self.data['Epoch'].max()
        final_data = self.data[self.data['Epoch'] == final_epoch]
        
        print(f"\nFinal Performance (Epoch {final_epoch}):")
        if len(final_data) > 0:
            print(f"Average Return: {final_data['AverageEpRet'].mean():.4f} ± {final_data['AverageEpRet'].std():.4f}")
            print(f"Max Return: {final_data['MaxEpRet'].mean():.4f} ± {final_data['MaxEpRet'].std():.4f}")
        print(f"Best Overall Return: {self.data['MaxEpRet'].max():.4f}")
        print(f"Number of seeds: {len(seeds)}")
        print(f"Total epochs: {len(self.data['Epoch'].unique())}")
        
        if save_to_file:
            fig.savefig('training_progress.png')
    
    def analyze_best_design(self, plot_spectrum=True, plot_angle_map=False):
        """Analyze the best design found"""
        if not self.best_designs:
            print("No designs found")
            return None
            
        best = self.best_designs[0]
        materials = best['materials']
        thicknesses = best['thicknesses']
        reward = best['reward']
        
        print(f"\nBest Design Analysis:")
        print(f"Materials: {materials}")
        print(f"Thicknesses: {thicknesses} nm")
        print(f"Reward: {reward:.6f}")
        
        if self.env is None:
            print("Environment not setup - cannot simulate spectrum")
            return best
            
        # Simulate spectrum
        thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
        R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
        
        if plot_spectrum:
            plt.figure(figsize=(10, 6))
            wavelengths = self.env.simulator.wavelength * 1000  # Convert to nm
            
            plt.plot(wavelengths, R, 'r-', label='Reflectance', linewidth=2)
            plt.plot(wavelengths, T, 'b-', label='Transmittance', linewidth=2)
            plt.plot(wavelengths, A, 'g-', label='Absorbance', linewidth=2)
            
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title(f'Best Design Spectrum (Reward: {reward:.4f})')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        if plot_angle_map:
            self._plot_angle_dependent_absorption(materials, thicknesses)
        
        return best
    
    def _plot_angle_dependent_absorption(self, materials, thicknesses, max_angle=70):
        """Plot angle-dependent absorption map"""
        print("Calculating angle-dependent absorption...")
        
        angles = np.arange(0, max_angle + 1, 2)  # Every 2 degrees
        thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
        
        all_absorptions = []
        for angle in tqdm(angles, desc="Calculating spectra"):
            try:
                R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, theta=angle, plot=False)
                all_absorptions.append(A)
            except Exception as e:
                print(f"Error at angle {angle}: {e}")
                # Use zeros as fallback
                wavelengths = self.env.simulator.wavelength
                all_absorptions.append(np.zeros_like(wavelengths))
        
        # Create 2D map
        absorption_map = np.array(all_absorptions).T
        wavelengths = self.env.simulator.wavelength * 1000  # nm
        
        plt.figure(figsize=(10, 6))
        im = plt.pcolormesh(angles, wavelengths, absorption_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, label='Absorption')
        plt.xlabel('Incidence Angle (degrees)')
        plt.ylabel('Wavelength (nm)')
        plt.title('Angle-Dependent Absorption')
        plt.show()
    
    def optimize_design(self, design_idx=0, max_thickness=200):
        """Optimize the best design using gradient-based optimization"""
        if not self.best_designs or self.env is None:
            print("No designs or environment available")
            return None
            
        design = self.best_designs[design_idx]
        materials = design['materials']
        initial_thicknesses = design['thicknesses']
        
        print(f"\nOptimizing design {design_idx}:")
        print(f"Initial: {materials} | {initial_thicknesses}")
        
        def objective(x):
            """Objective function to maximize reward"""
            thickness_with_inf = [float('inf')] + list(x) + [float('inf')]
            R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
            from RLMultilayer.utils import cal_reward
            reward = cal_reward(R, T, A, self.env.target)
            return 1 - reward  # Minimize negative reward
        
        # Set bounds
        bounds = [(15, max_thickness)] * len(initial_thicknesses)
        
        # Optimize
        result = minimize(objective, initial_thicknesses, bounds=bounds, method='L-BFGS-B')
        
        optimized_thicknesses = [int(x) for x in result.x]
        optimized_reward = 1 - result.fun
        improvement = optimized_reward - design['reward']
        
        print(f"Optimized: {materials} | {optimized_thicknesses}")
        print(f"Initial reward: {design['reward']:.6f}")
        print(f"Optimized reward: {optimized_reward:.6f}")
        print(f"Improvement: {improvement:.6f}")
        
        return {
            'materials': materials,
            'thicknesses': optimized_thicknesses,
            'reward': optimized_reward,
            'improvement': improvement
        }
    
    def compare_designs(self, num_designs=5):
        """Compare multiple top designs"""
        if len(self.best_designs) < num_designs:
            num_designs = len(self.best_designs)
            
        print(f"\nComparing top {num_designs} designs:")
        
        rewards = []
        labels = []
        
        for i in range(num_designs):
            design = self.best_designs[i]
            rewards.append(design['reward'])
            labels.append(f"Design {i+1}")
            print(f"{i+1}. {design['materials']} | {design['thicknesses']} | {design['reward']:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, rewards)
        plt.ylabel('Reward')
        plt.title('Top Designs Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    def full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 60)
        print("OPTICAL DESIGN ANALYSIS PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return
            
        # Step 2: Setup environment
        if not self.setup_environment():
            print("Continuing without environment (limited functionality)")
            
        # Step 3: Plot training progress
        print("\n" + "="*40)
        print("TRAINING PROGRESS")
        print("="*40)
        self.plot_training_progress()
        
        # Step 4: Analyze best design
        print("\n" + "="*40)
        print("BEST DESIGN ANALYSIS")
        print("="*40)
        best = self.analyze_best_design(plot_spectrum=True, plot_angle_map=False)
        
        # Step 5: Compare top designs
        print("\n" + "="*40)
        print("DESIGN COMPARISON")
        print("="*40)
        self.compare_designs(num_designs=min(5, len(self.best_designs)))
        
        # Step 6: Optimization
        if self.env is not None:
            print("\n" + "="*40)
            print("DESIGN OPTIMIZATION")
            print("="*40)
            optimized = self.optimize_design()
            
            if optimized and optimized['improvement'] > 0.001:
                print("\nPlotting optimized design spectrum...")
                self.analyze_best_design_from_dict(optimized, plot_spectrum=True)
                
                # Plot original vs optimized side by side
                original_design_dict = {
                    'materials': best['materials'],
                    'thicknesses': best['thicknesses'],
                    'reward': best['reward']
                }
                self.plot_original_vs_optimized_spectra(original_design_dict, optimized)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return best
    
    def analyze_best_design_from_dict(self, design_dict, plot_spectrum=True):
        """Analyze a design from dictionary format"""
        materials = design_dict['materials']
        thicknesses = design_dict['thicknesses']
        reward = design_dict['reward']
        
        if plot_spectrum and self.env is not None:
            thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
            R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
            
            plt.figure(figsize=(10, 6))
            wavelengths = self.env.simulator.wavelength * 1000
            
            plt.plot(wavelengths, R, 'r-', label='Reflectance', linewidth=2)
            plt.plot(wavelengths, T, 'b-', label='Transmittance', linewidth=2)
            plt.plot(wavelengths, A, 'g-', label='Absorbance', linewidth=2)
            
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title(f'Optimized Design Spectrum (Reward: {reward:.4f})')
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_original_vs_optimized_spectra(self, original_design, optimized_design):
        """Plot original and optimized spectra side by side"""
        if self.env is None:
            print("Environment not setup - cannot plot spectra.")
            return

        print("\nPlotting original vs. optimized spectra side by side...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        wavelengths = self.env.simulator.wavelength * 1000 # nm

        # Plot Original Design
        orig_materials = original_design['materials']
        orig_thicknesses = original_design['thicknesses']
        orig_reward = original_design['reward']
        
        orig_thick_inf = [float('inf')] + orig_thicknesses + [float('inf')]
        R_orig, T_orig, A_orig = self.env.simulator.spectrum(orig_materials, orig_thick_inf, plot=False)

        axes[0].plot(wavelengths, R_orig, 'r-', label='Reflectance', linewidth=2)
        axes[0].plot(wavelengths, T_orig, 'b-', label='Transmittance', linewidth=2)
        axes[0].plot(wavelengths, A_orig, 'g-', label='Absorbance', linewidth=2)
        axes[0].set_title(f'Original Design (Reward: {orig_reward:.4f})')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Intensity')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Optimized Design
        opt_materials = optimized_design['materials']
        opt_thicknesses = optimized_design['thicknesses']
        opt_reward = optimized_design['reward']

        opt_thick_inf = [float('inf')] + opt_thicknesses + [float('inf')]
        R_opt, T_opt, A_opt = self.env.simulator.spectrum(opt_materials, opt_thick_inf, plot=False)

        axes[1].plot(wavelengths, R_opt, 'r-', label='Reflectance', linewidth=2)
        axes[1].plot(wavelengths, T_opt, 'b-', label='Transmittance', linewidth=2)
        axes[1].plot(wavelengths, A_opt, 'g-', label='Absorbance', linewidth=2)
        axes[1].set_title(f'Optimized Design (Reward: {opt_reward:.4f})')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Intensity')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    def get_numerical_spectrum(self, design_idx=0, wavelength_range=(400, 2000), num_points=50):
        """Get numerical representation of spectrum for LLM analysis"""
        if not self.best_designs or self.env is None:
            print("No designs or environment available")
            return None
            
        design = self.best_designs[design_idx]
        materials = design['materials']
        thicknesses = design['thicknesses']
        reward = design['reward']
        
        print(f"\nNumerical Spectrum Analysis - Design {design_idx}:")
        print(f"Materials: {materials}")
        print(f"Thicknesses: {thicknesses} nm")
        print(f"Reward: {reward:.6f}")
        print("="*80)
        
        # Calculate spectrum
        thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
        R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
        wavelengths = self.env.simulator.wavelength * 1000  # Convert to nm
        
        # Sample at specified points for easier LLM consumption
        if len(wavelengths) > num_points:
            indices = np.linspace(0, len(wavelengths)-1, num_points, dtype=int)
            sampled_wavelengths = wavelengths[indices]
            sampled_R = R[indices]
            sampled_T = T[indices]
            sampled_A = A[indices]
        else:
            sampled_wavelengths = wavelengths
            sampled_R = R
            sampled_T = T
            sampled_A = A
        
        # Create structured output
        spectrum_data = {
            'design': {
                'materials': materials,
                'thicknesses': thicknesses,
                'reward': reward,
                'num_layers': len(materials)
            },
            'spectrum': {
                'wavelength_nm': sampled_wavelengths.tolist(),
                'reflectance': sampled_R.tolist(),
                'transmittance': sampled_T.tolist(),
                'absorbance': sampled_A.tolist()
            },
            'key_metrics': self._calculate_key_metrics(wavelengths, R, T, A)
        }
        
        # Print formatted table for LLM
        print("WAVELENGTH-DEPENDENT OPTICAL PROPERTIES:")
        print(f"{'Wavelength(nm)':<12} {'Reflectance':<12} {'Transmittance':<14} {'Absorbance':<12}")
        print("-" * 60)
        
        for i in range(len(sampled_wavelengths)):
            wl = sampled_wavelengths[i]
            r = sampled_R[i]
            t = sampled_T[i]
            a = sampled_A[i]
            print(f"{wl:<12.0f} {r:<12.4f} {t:<14.4f} {a:<12.4f}")
        
        print("\nKEY PERFORMANCE METRICS:")
        for metric, value in spectrum_data['key_metrics'].items():
            print(f"{metric}: {value}")
        
        print("\nDESIGN SUMMARY FOR LLM:")
        print(f"This is a {len(materials)}-layer optical absorber with materials {materials}")
        print(f"and thicknesses {thicknesses} nm respectively.")
        print(f"Performance reward: {reward:.6f}")
        
        target_wl = 1500  # Target wavelength for absorber
        closest_idx = np.argmin(np.abs(wavelengths - target_wl))
        target_abs = A[closest_idx]
        print(f"Absorption at target wavelength ({target_wl}nm): {target_abs:.4f} ({target_abs*100:.1f}%)")
        
        return spectrum_data
    
    def _calculate_key_metrics(self, wavelengths, R, T, A):
        """Calculate key optical performance metrics"""
        # Find target wavelength (1500nm) performance
        target_wl = 1500
        closest_idx = np.argmin(np.abs(wavelengths - target_wl))
        
        # Calculate average values in different bands
        vis_mask = (wavelengths >= 400) & (wavelengths <= 700)  # Visible
        nir_mask = (wavelengths >= 700) & (wavelengths <= 1400)  # Near-IR
        target_mask = (wavelengths >= 1400) & (wavelengths <= 1600)  # Target band around 1500nm
        
        metrics = {
            'target_absorption_1500nm': float(A[closest_idx]),
            'avg_absorption_visible': float(np.mean(A[vis_mask])) if np.any(vis_mask) else 0,
            'avg_absorption_near_ir': float(np.mean(A[nir_mask])) if np.any(nir_mask) else 0,
            'avg_absorption_target_band': float(np.mean(A[target_mask])) if np.any(target_mask) else 0,
            'peak_absorption': float(np.max(A)),
            'peak_absorption_wavelength': float(wavelengths[np.argmax(A)]),
            'min_reflectance': float(np.min(R)),
            'avg_transmittance': float(np.mean(T)),
            'bandwidth_80_percent': self._calculate_bandwidth(wavelengths, A, 0.8)
        }
        
        return metrics
    
    def _calculate_bandwidth(self, wavelengths, absorption, threshold):
        """Calculate bandwidth where absorption > threshold"""
        above_threshold = absorption > threshold
        if not np.any(above_threshold):
            return 0
        
        indices = np.where(above_threshold)[0]
        if len(indices) < 2:
            return 0
            
        bandwidth = wavelengths[indices[-1]] - wavelengths[indices[0]]
        return float(bandwidth)
    def get_best_designs_with_spectra(self, num_designs=5, show_plots=True, save_data=True):
        """Get best designs with their spectrum graphs and numerical data"""
        if not self.best_designs or self.env is None:
            print("No designs or environment available")
            return None
            
        num_designs = min(num_designs, len(self.best_designs))
        
        print("=" * 80)
        print(f"BEST {num_designs} OPTICAL ABSORBER DESIGNS WITH SPECTRA")
        print("=" * 80)
        
        all_design_data = []
        
        # Create subplots for all designs
        if show_plots:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten() if num_designs > 1 else [axes]
        
        for i in range(num_designs):
            design = self.best_designs[i]
            materials = design['materials']
            thicknesses = design['thicknesses']
            reward = design['reward']
            
            print(f"\n{'='*20} DESIGN {i+1} {'='*20}")
            print(f"Materials: {materials}")
            print(f"Thicknesses: {thicknesses} nm")
            print(f"Reward: {reward:.6f}")
            print(f"Number of layers: {len(materials)}")
            
            # Calculate spectrum
            thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
            R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
            wavelengths = self.env.simulator.wavelength * 1000  # Convert to nm
            
            # Calculate key metrics
            key_metrics = self._calculate_key_metrics(wavelengths, R, T, A)
            
            # Print key metrics
            print(f"\nKey Performance Metrics:")
            print(f"- Absorption at 1500nm: {key_metrics['target_absorption_1500nm']:.4f} ({key_metrics['target_absorption_1500nm']*100:.1f}%)")
            print(f"- Peak absorption: {key_metrics['peak_absorption']:.4f} at {key_metrics['peak_absorption_wavelength']:.0f}nm")
            print(f"- Average absorption (1400-1600nm): {key_metrics['avg_absorption_target_band']:.4f}")
            print(f"- Bandwidth (>80% absorption): {key_metrics['bandwidth_80_percent']:.0f}nm")
            
            # Store design data
            design_data = {
                'rank': i + 1,
                'materials': materials,
                'thicknesses': thicknesses,
                'reward': reward,
                'wavelengths': wavelengths.tolist(),
                'reflectance': R.tolist(),
                'transmittance': T.tolist(),
                'absorbance': A.tolist(),
                'key_metrics': key_metrics
            }
            all_design_data.append(design_data)
            
            # Plot spectrum
            if show_plots and i < 6:  # Limit to 6 plots
                ax = axes[i] if num_designs > 1 else axes[0]
                
                ax.plot(wavelengths, R, 'r-', label='Reflectance', linewidth=2, alpha=0.8)
                ax.plot(wavelengths, T, 'b-', label='Transmittance', linewidth=2, alpha=0.8)
                ax.plot(wavelengths, A, 'g-', label='Absorbance', linewidth=2, alpha=0.8)
                
                # Highlight target wavelength
                ax.axvline(x=1500, color='purple', linestyle='--', alpha=0.7, label='Target (1500nm)')
                
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.set_title(f'Design {i+1}: {" + ".join(materials[:2])}{"..." if len(materials)>2 else ""}\nReward: {reward:.4f}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(400, 2000)
                ax.set_ylim(0, 1)
        
        if show_plots:
            # Hide unused subplots
            for j in range(num_designs, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE OF BEST DESIGNS:")
        print(f"{'='*80}")
        print(f"{'Rank':<4} {'Materials':<25} {'Thicknesses':<20} {'Reward':<8} {'1500nm Abs':<10}")
        print("-" * 80)
        
        for data in all_design_data:
            materials_str = " + ".join(data['materials'][:3])
            if len(data['materials']) > 3:
                materials_str += "..."
            thicknesses_str = str(data['thicknesses'][:3])
            if len(data['thicknesses']) > 3:
                thicknesses_str = thicknesses_str[:-1] + "...]"
            
            print(f"{data['rank']:<4} {materials_str:<25} {thicknesses_str:<20} {data['reward']:<8.4f} {data['key_metrics']['target_absorption_1500nm']:<10.3f}")
        
        # Save data if requested
        if save_data:
            output_file = 'best_designs_with_spectra.json'
            with open(output_file, 'w') as f:
                json.dump(all_design_data, f, indent=2, cls=NumpyEncoder)
            print(f"\nData saved to: {output_file}")
        
        return all_design_data
    
    def plot_best_designs_comparison(self, num_designs=3):
        """Plot comparison of best designs side by side"""
        if not self.best_designs or self.env is None:
            print("No designs or environment available")
            return
            
        num_designs = min(num_designs, len(self.best_designs))
        
        fig, axes = plt.subplots(1, num_designs, figsize=(6*num_designs, 6))
        if num_designs == 1:
            axes = [axes]
            
        for i in range(num_designs):
            design = self.best_designs[i]
            materials = design['materials']
            thicknesses = design['thicknesses']
            reward = design['reward']
            
            # Calculate spectrum
            thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
            R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
            wavelengths = self.env.simulator.wavelength * 1000
            
            # Plot
            axes[i].plot(wavelengths, R, 'r-', label='Reflectance', linewidth=2)
            axes[i].plot(wavelengths, T, 'b-', label='Transmittance', linewidth=2)
            axes[i].plot(wavelengths, A, 'g-', label='Absorbance', linewidth=2)
            axes[i].axvline(x=1500, color='purple', linestyle='--', alpha=0.7)
            
            axes[i].set_xlabel('Wavelength (nm)')
            axes[i].set_ylabel('Intensity')
            axes[i].set_title(f'Design {i+1} (Reward: {reward:.4f})\n{materials}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(400, 2000)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def export_spectrum_data_csv(self, design_idx=0, filename=None):
        """Export spectrum data to CSV for external analysis"""
        if not self.best_designs or self.env is None:
            print("No designs or environment available")
            return
            
        design = self.best_designs[design_idx]
        materials = design['materials']
        thicknesses = design['thicknesses']
        reward = design['reward']
        
        # Calculate spectrum
        thickness_with_inf = [float('inf')] + thicknesses + [float('inf')]
        R, T, A = self.env.simulator.spectrum(materials, thickness_with_inf, plot=False)
        wavelengths = self.env.simulator.wavelength * 1000
        
        # Create DataFrame
        df = pd.DataFrame({
            'Wavelength_nm': wavelengths,
            'Reflectance': R,
            'Transmittance': T,
            'Absorbance': A
        })
        
        if filename is None:
            materials_str = "_".join(materials)
            filename = f"design_{design_idx+1}_{materials_str}_spectrum.csv"
        
        df.to_csv(filename, index=False)
        print(f"Spectrum data exported to: {filename}")
        print(f"Design: {materials} | {thicknesses} nm | Reward: {reward:.6f}")
        
        return df

    def generate_pdf_report(self, best_designs_data, filename="Comprehensive_Optical_Design_Report.pdf"):
        """Generate a comprehensive PDF report using matplotlib PdfPages"""
        
        print(f"Generating PDF report: {filename}...")
        
        with PdfPages(filename) as pdf:
            # Title Page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.9, 'Comprehensive Optical Design Analysis Report', 
                   transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center')
            ax.text(0.5, 0.8, f'Experiment: {os.path.basename(self.experiment_folder)}', 
                   transform=ax.transAxes, fontsize=14, ha='center')
            ax.text(0.5, 0.75, f'Number of designs analyzed: {len(best_designs_data)}', 
                   transform=ax.transAxes, fontsize=12, ha='center')
            ax.text(0.5, 0.7, f'Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                   transform=ax.transAxes, fontsize=10, ha='center')
            
            # Executive Summary
            summary_text = (
                f"This report summarizes the results of an RL-based optical design experiment.\n\n"
                f"Key Findings:\n"
                f"• Best design reward: {best_designs_data[0]['reward']:.6f}\n"
                f"• Target absorption at 1500nm: {best_designs_data[0]['key_metrics']['target_absorption_1500nm']:.3f}\n"
                f"• Peak absorption: {best_designs_data[0]['key_metrics']['peak_absorption']:.3f}\n"
                f"• Materials used: {', '.join(set([mat for design in best_designs_data for mat in design['materials']]))}"
            )
            ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top', wrap=True)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Training Progress
            if hasattr(self, 'data') and self.data is not None:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                seeds = self.data['seed'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(seeds)))
                
                # Average Return
                for i, seed in enumerate(seeds):
                    seed_data = self.data[self.data['seed'] == seed]
                    axes[0,0].plot(seed_data['Epoch'], seed_data['AverageEpRet'], 
                                  color=colors[i], alpha=0.7, linewidth=1.5)
                axes[0,0].set_title('Average Episode Return', fontsize=14, fontweight='bold')
                axes[0,0].set_xlabel('Epoch')
                axes[0,0].set_ylabel('Average Return')
                axes[0,0].grid(True)
                
                # Max Return
                for i, seed in enumerate(seeds):
                    seed_data = self.data[self.data['seed'] == seed]
                    axes[0,1].plot(seed_data['Epoch'], seed_data['MaxEpRet'], 
                                  color=colors[i], alpha=0.7, linewidth=1.5)
                axes[0,1].set_title('Maximum Episode Return', fontsize=14, fontweight='bold')
                axes[0,1].set_xlabel('Epoch')
                axes[0,1].set_ylabel('Max Return')
                axes[0,1].grid(True)
                
                # Episode Length
                if 'EpLen' in self.data.columns:
                    for i, seed in enumerate(seeds):
                        seed_data = self.data[self.data['seed'] == seed]
                        axes[1,0].plot(seed_data['Epoch'], seed_data['EpLen'], 
                                      color=colors[i], alpha=0.7, linewidth=1.5)
                    axes[1,0].set_title('Episode Length', fontsize=14, fontweight='bold')
                    axes[1,0].set_xlabel('Epoch')
                    axes[1,0].set_ylabel('Episode Length')
                    axes[1,0].grid(True)
                else:
                    axes[1,0].text(0.5, 0.5, 'EpLen not available', 
                                  transform=axes[1,0].transAxes, ha='center', va='center')
                    axes[1,0].set_title('Episode Length (N/A)', fontsize=14, fontweight='bold')
                
                # Entropy
                if 'Entropy' in self.data.columns:
                    for i, seed in enumerate(seeds):
                        seed_data = self.data[self.data['seed'] == seed]
                        axes[1,1].plot(seed_data['Epoch'], seed_data['Entropy'], 
                                      color=colors[i], alpha=0.7, linewidth=1.5)
                    axes[1,1].set_title('Policy Entropy', fontsize=14, fontweight='bold')
                    axes[1,1].set_xlabel('Epoch')
                    axes[1,1].set_ylabel('Entropy')
                    axes[1,1].grid(True)
                else:
                    axes[1,1].text(0.5, 0.5, 'Entropy not available', 
                                  transform=axes[1,1].transAxes, ha='center', va='center')
                    axes[1,1].set_title('Policy Entropy (N/A)', fontsize=14, fontweight='bold')
                
                plt.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Individual Design Analysis Pages
            for i, design_data in enumerate(best_designs_data):
                fig = plt.figure(figsize=(11, 8.5))
                
                # Create main spectrum plot
                ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
                
                wavelengths = np.array(design_data['wavelengths'])
                ax1.plot(wavelengths, design_data['reflectance'], 'r-', label='Reflectance', linewidth=2.5)
                ax1.plot(wavelengths, design_data['transmittance'], 'b-', label='Transmittance', linewidth=2.5)
                ax1.plot(wavelengths, design_data['absorbance'], 'g-', label='Absorbance', linewidth=2.5)
                ax1.axvline(x=1500, color='purple', linestyle='--', alpha=0.7, label='Target (1500nm)')
                
                ax1.set_xlabel('Wavelength (nm)', fontsize=12)
                ax1.set_ylabel('Intensity', fontsize=12)
                ax1.set_title(f'Design #{design_data["rank"]} Spectrum (Reward: {design_data["reward"]:.4f})', 
                             fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(400, 2000)
                ax1.set_ylim(0, 1.05)
                
                # Design parameters text box
                ax2 = plt.subplot2grid((3, 2), (2, 0))
                ax2.axis('off')
                
                param_text = (
                    f"Materials: {design_data['materials']}\n"
                    f"Thicknesses (nm): {design_data['thicknesses']}\n"
                    f"Number of layers: {len(design_data['materials'])}\n"
                    f"Reward: {design_data['reward']:.6f}"
                )
                
                ax2.text(0.05, 0.95, "Design Parameters:", fontsize=12, fontweight='bold', 
                        transform=ax2.transAxes, verticalalignment='top')
                ax2.text(0.05, 0.75, param_text, fontsize=10, 
                        transform=ax2.transAxes, verticalalignment='top')
                
                # Key metrics text box
                ax3 = plt.subplot2grid((3, 2), (2, 1))
                ax3.axis('off')
                
                metrics = design_data['key_metrics']
                metrics_text = (
                    f"1500nm Absorption: {metrics['target_absorption_1500nm']:.4f}\n"
                    f"Peak Absorption: {metrics['peak_absorption']:.4f}\n"
                    f"Peak at: {metrics['peak_absorption_wavelength']:.0f} nm\n"
                    f"Target Band Avg: {metrics['avg_absorption_target_band']:.4f}\n"
                    f"80% Bandwidth: {metrics['bandwidth_80_percent']:.0f} nm"
                )
                
                ax3.text(0.05, 0.95, "Key Metrics:", fontsize=12, fontweight='bold', 
                        transform=ax3.transAxes, verticalalignment='top')
                ax3.text(0.05, 0.75, metrics_text, fontsize=10, 
                        transform=ax3.transAxes, verticalalignment='top')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Summary comparison page
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            
            # Rewards comparison
            ranks = [design['rank'] for design in best_designs_data]
            rewards = [design['reward'] for design in best_designs_data]
            
            bars = ax1.bar(ranks, rewards, color='skyblue', alpha=0.7, edgecolor='navy')
            ax1.set_xlabel('Design Rank')
            ax1.set_ylabel('Reward')
            ax1.set_title('Reward Comparison Across Best Designs', fontsize=14, fontweight='bold')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, reward in zip(bars, rewards):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{reward:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 1500nm absorption comparison
            absorptions_1500 = [design['key_metrics']['target_absorption_1500nm'] for design in best_designs_data]
            
            bars2 = ax2.bar(ranks, absorptions_1500, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax2.set_xlabel('Design Rank')
            ax2.set_ylabel('Absorption at 1500nm')
            ax2.set_title('Target Wavelength Performance Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, abs_val in zip(bars2, absorptions_1500):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{abs_val:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Successfully generated PDF report: {filename}")
        print(f"Report contains {len(best_designs_data) + 3} pages:")
        print("- Title and summary page")
        print("- Training progress analysis")
        print(f"- {len(best_designs_data)} individual design analysis pages")
        print("- Summary comparison page")

# %%
# Run the complete analysis

analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
best_design = analyzer.full_analysis()
















# %%
# Optional: Individual analysis steps
# Uncomment any of these for focused analysis

# Just load and plot training data
analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.plot_training_progress()
# Just analyze best design with angle map
analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.setup_environment()
analyzer.analyze_best_design(plot_spectrum=True, plot_angle_map=True)
# Just optimization
analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.setup_environment()
optimized = analyzer.optimize_design() 
# %%
# side by side plot
analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.setup_environment()
analyzer.plot_original_vs_optimized_spectra(best_design, optimized)
# %%
 
# %%
#get best design spectrum as discrete numbers to give llm to understand

# ... existing code ...


# %%

# %%
#get best design spectrum as discrete numbers to give llm to understand
analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.setup_environment()

# Get numerical spectrum data
spectrum_data = analyzer.get_numerical_spectrum(design_idx=0, num_points=100)

# Also get top 3 designs for comparison
print("\n" + "="*80)
print("COMPARISON OF TOP 3 DESIGNS:")
print("="*80)

for i in range(min(3, len(analyzer.best_designs))):
    print(f"\n--- DESIGN {i+1} ---")
    analyzer.get_numerical_spectrum(design_idx=i, num_points=100)
# %%
# I want to get best designs and their spectrum graphs. 

# MAIN ANALYSIS: Get Best Designs with Spectra
print("Starting comprehensive analysis of best optical absorber designs...")

analyzer = OpticalDesignAnalyzer(EXPERIMENT_FOLDER, ENV_NAME)
analyzer.load_data()
analyzer.setup_environment()

# Get best designs with spectra (both plots and numerical data)
best_designs_data = analyzer.get_best_designs_with_spectra(
    num_designs=5, 
    show_plots=True, 
    save_data=True
)

# Plot side-by-side comparison of top 3
print("\nCreating side-by-side comparison...")
analyzer.plot_best_designs_comparison(num_designs=3)

# Export individual design data
print("\nExporting individual design data...")
for i in range(min(3, len(analyzer.best_designs))):
    analyzer.export_spectrum_data_csv(design_idx=i)

# Get detailed numerical data for LLM analysis
print("\nDetailed numerical analysis for LLM:")
for i in range(min(3, len(analyzer.best_designs))):
    print(f"\n--- DESIGN {i+1} NUMERICAL DATA ---")
    spectrum_data = analyzer.get_numerical_spectrum(design_idx=i, num_points=20)

# Generate the final PDF report
if best_designs_data:
    analyzer.generate_pdf_report(best_designs_data)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("Files generated:")
print("- best_designs_with_spectra.json (comprehensive data)")
print("- design_*_spectrum.csv (individual spectrum data)")
print("- Comprehensive_Optical_Design_Report.pdf (PDF report)")
print("="*80)
# %%

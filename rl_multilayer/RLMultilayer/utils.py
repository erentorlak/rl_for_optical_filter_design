from mpi4py import MPI
import matplotlib
from tmm import coh_tmm
import pandas as pd
import os
from numpy import pi
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from scipy.optimize import minimize
import json
from tqdm import tqdm

# Data directory - simple approach since running from project root
DATABASE = 'data'
INSULATORS = ['HfO2', 'SiO2', 'SiC', 'Al2O3', 'MgF2', 'TiO2', 'Fe2O3', 'MgF2', 'Si3N4', 'TiN', 'ZnO', 'ZnS', 'ZnSe']
METALS = ['Ag', 'Al', 'Cr', 'Ge', 'Si', 'Ni']

# SiO2, TiO2,
# glass, PECVD-Si, Sputter-Sİ, Ti


num_workers = 8

def cal_reward(R, T, A, target):
    '''
    Calculate reward based on given spectrums. 
    We calculate the reward using averaged (1-mse).

    Args:
        R, T, A: numpy array. Reflection, transmission, and 
        absorption spectrums, respectively.
        target: dict. {'R':np.array, 'T':np.array, 'A':np.array}

    Returns:
        reward: float. Reward for the spectrum. 
    '''

    reward = 0
    for k, v in target.items():

        if k == 'R':
            res = R
        elif k == 'T':
            res = T
        else:
            res = A
        
        reward += 1 - np.abs(res.squeeze() - v).mean()

    reward /= len(target)

    return reward


def cal_reward_selective_1500(R, T, A, target, wavelengths=None, target_wavelength=1.5):
    '''
    Enhanced reward function for 1500nm selective absorber tasks.
    
    This function is specifically designed for selective absorber applications where:
    1. Perfect absorption (A=1) is needed in 1450-1550nm range
    2. High reflection (R=1) is desired everywhere else  
    3. Zero transmission (T=0) across the entire spectrum
    4. Peak sharpness and selectivity are rewarded
    
    Args:
        R, T, A: numpy arrays. Reflection, transmission, and absorption spectra
        target: dict. Target spectra {'R': array, 'T': array, 'A': array}
        wavelengths: numpy array. Wavelength points (in μm)
        target_wavelength: float. Center wavelength for selective absorption (in μm)
        
    Returns:
        reward: float. Enhanced reward emphasizing selective behavior
    '''
    
    # Base reward using MAE for spectrum matching
    base_reward = 0
    for k, v in target.items():
        if k == 'R':
            res = R
        elif k == 'T':
            res = T
        else:
            res = A
            
        # Use Mean Absolute Error (consistent with original)
        mae = np.abs(res.squeeze() - v).mean()
        base_reward += 1 - mae
        
    base_reward /= len(target)
    
    # Enhanced reward components for selective behavior
    enhancement = 0
    
    if wavelengths is not None:
        # 1. Transmission penalty - heavily penalize any transmission anywhere
        T_penalty = -15.0 * np.mean(T**2)  # Strong quadratic penalty for transmission
        
        # 2. Define absorption window (1450-1550nm = 1.45-1.55μm)
        absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
        reflection_window = ~absorption_window
        
        if np.any(absorption_window) and np.any(reflection_window):
            # Peak absorption strength in target window
            peak_absorption = np.mean(A[absorption_window])
            
            # Off-peak reflection strength (should be high for selectivity)
            off_peak_reflection = np.mean(R[reflection_window])
            
            # Off-peak absorption (should be low for selectivity)
            off_peak_absorption = np.mean(A[reflection_window])
            
            # Transmission in both windows (should be zero everywhere)
            peak_transmission = np.mean(T[absorption_window])
            off_peak_transmission = np.mean(T[reflection_window])
            
            # 3. Selectivity bonus - reward sharp distinction between absorption and reflection regions
            selectivity = peak_absorption - off_peak_absorption
            selectivity_bonus = 5.0 * np.clip(selectivity, 0, 1)
            
            # 4. Peak quality bonus - reward strong absorption in target window
            peak_bonus = 3.0 * peak_absorption
            
            # 5. Off-peak reflection bonus - reward high reflection away from peak
            # BUT heavily penalize if there's NO absorption in target window
            if peak_absorption < 0.1:  # If almost no absorption in target window
                reflection_bonus = -5.0 * off_peak_reflection  # Penalize pure reflectors
            else:
                reflection_bonus = 2.0 * off_peak_reflection
            
            # 6. Transmission penalties for both regions
            peak_T_penalty = -10.0 * peak_transmission**2
            off_peak_T_penalty = -8.0 * off_peak_transmission**2
            
            # 7. Edge sharpness bonus - reward sharp transitions at boundaries
            if len(wavelengths) > 10:  # Ensure enough points for gradient calculation
                absorption_gradient = np.abs(np.gradient(A))
                edge_indices = np.where((wavelengths >= 1.44) & (wavelengths <= 1.46) | 
                                      (wavelengths >= 1.54) & (wavelengths <= 1.56))[0]
                if len(edge_indices) > 0:
                    edge_sharpness = np.mean(absorption_gradient[edge_indices])
                    sharpness_bonus = 1.5 * np.clip(edge_sharpness, 0, 2)
                else:
                    sharpness_bonus = 0
            else:
                sharpness_bonus = 0
            
            # 8. Uniformity bonus within absorption window
            absorption_uniformity = 1.0 - np.std(A[absorption_window])
            uniformity_bonus = 1.0 * np.clip(absorption_uniformity, 0, 1)
            
            enhancement = (T_penalty + selectivity_bonus + peak_bonus + reflection_bonus + 
                         peak_T_penalty + off_peak_T_penalty + sharpness_bonus + uniformity_bonus)
        else:
            # Fallback if wavelength indexing fails
            enhancement = T_penalty
    else:
        # If no wavelength info, just penalize transmission
        enhancement = -10.0 * np.mean(T**2)
    
    # Final reward combines base matching with selective behavior enhancements
    final_reward = base_reward + 0.3 * enhancement  # Weight enhancement at 30%
    
    return final_reward


def cal_reward_selective_configurable(R, T, A, target, wavelengths=None, 
                                    target_wavelength=1.5, window_width=0.05, 
                                    enhancement_weight=0.3, penalty_weights=None):
    '''
    Configurable reward function for selective absorber tasks at any wavelength.
    
    This function generalizes the selective absorber reward to work with any target
    wavelength and absorption window width. It maintains the same reward structure
    as cal_reward_selective_1500 but allows flexible configuration.
    
    Args:
        R, T, A: numpy arrays. Reflection, transmission, and absorption spectra
        target: dict. Target spectra {'R': array, 'T': array, 'A': array}
        wavelengths: numpy array. Wavelength points (in μm)
        target_wavelength: float. Center wavelength for selective absorption (in μm)
        window_width: float. Half-width of absorption window (in μm). 
                     E.g., 0.05 = ±50nm window around target
        enhancement_weight: float. Weight for enhancement terms vs base reward (0-1)
        penalty_weights: dict. Custom penalty weights for fine-tuning behavior
                        Default: {'transmission': 15.0, 'selectivity': 5.0, 'peak': 3.0, 
                                'reflection': 2.0, 'peak_transmission': 10.0, 
                                'off_peak_transmission': 8.0, 'sharpness': 1.5, 'uniformity': 1.0}
        
    Returns:
        reward: float. Enhanced reward emphasizing selective behavior at target wavelength
        
    Examples:
        # 1550nm telecom absorber with ±50nm window
        reward = cal_reward_selective_configurable(R, T, A, target, wavelengths, 
                                                 target_wavelength=1.55, window_width=0.05)
        
        # 980nm NIR absorber with ±40nm window  
        reward = cal_reward_selective_configurable(R, T, A, target, wavelengths,
                                                 target_wavelength=0.98, window_width=0.04)
        
        # Custom penalty weights for aggressive transmission suppression
        custom_penalties = {'transmission': 25.0, 'peak_transmission': 15.0}
        reward = cal_reward_selective_configurable(R, T, A, target, wavelengths,
                                                 target_wavelength=1.3, window_width=0.06,
                                                 penalty_weights=custom_penalties)
    '''
    
    # Default penalty weights
    default_penalties = {
        'transmission': 15.0,           # Global transmission penalty
        'selectivity': 5.0,             # Selectivity bonus weight
        'peak': 3.0,                    # Peak absorption bonus
        'reflection': 2.0,              # Off-peak reflection bonus
        'peak_transmission': 10.0,      # Peak region transmission penalty
        'off_peak_transmission': 8.0,   # Off-peak region transmission penalty
        'sharpness': 1.5,               # Edge sharpness bonus
        'uniformity': 1.0,              # Absorption uniformity bonus
        'anti_broadband': 5.0           # Penalty for broadband absorbers/reflectors
    }
    
    # Merge custom penalty weights with defaults
    if penalty_weights is None:
        penalty_weights = default_penalties
    else:
        penalties = default_penalties.copy()
        penalties.update(penalty_weights)
        penalty_weights = penalties
    
    # Base reward using MAE for spectrum matching
    base_reward = 0
    for k, v in target.items():
        if k == 'R':
            res = R
        elif k == 'T':
            res = T
        else:
            res = A
            
        # Use Mean Absolute Error (consistent with original)
        mae = np.abs(res.squeeze() - v).mean()
        base_reward += 1 - mae
        
    base_reward /= len(target)
    
    # Enhanced reward components for selective behavior
    enhancement = 0
    
    if wavelengths is not None:
        # 1. Global transmission penalty - heavily penalize any transmission anywhere
        T_penalty = -penalty_weights['transmission'] * np.mean(T**2)
        
        # 2. Define absorption and reflection windows based on target wavelength
        window_min = target_wavelength - window_width
        window_max = target_wavelength + window_width
        absorption_window = (wavelengths >= window_min) & (wavelengths <= window_max)
        reflection_window = ~absorption_window
        
        if np.any(absorption_window) and np.any(reflection_window):
            # Peak absorption strength in target window
            peak_absorption = np.mean(A[absorption_window])
            
            # Off-peak reflection strength (should be high for selectivity)
            off_peak_reflection = np.mean(R[reflection_window])
            
            # Off-peak absorption (should be low for selectivity)
            off_peak_absorption = np.mean(A[reflection_window])
            
            # Transmission in both windows (should be zero everywhere)
            peak_transmission = np.mean(T[absorption_window])
            off_peak_transmission = np.mean(T[reflection_window])
            
            # 3. Selectivity bonus - reward sharp distinction between absorption and reflection regions
            selectivity = peak_absorption - off_peak_absorption
            selectivity_bonus = penalty_weights['selectivity'] * np.clip(selectivity, 0, 1)
            
            # 4. Peak quality bonus - reward strong absorption in target window
            peak_bonus = penalty_weights['peak'] * peak_absorption
            
            # 5. Off-peak reflection bonus - reward high reflection away from peak
            # BUT heavily penalize if there's NO absorption in target window (broadband reflector)
            if peak_absorption < 0.1:  # If almost no absorption in target window
                reflection_bonus = -penalty_weights['anti_broadband'] * off_peak_reflection  # Penalize pure reflectors
            else:
                reflection_bonus = penalty_weights['reflection'] * off_peak_reflection
            
            # 6. Transmission penalties for both regions
            peak_T_penalty = -penalty_weights['peak_transmission'] * peak_transmission**2
            off_peak_T_penalty = -penalty_weights['off_peak_transmission'] * off_peak_transmission**2
            
            # 7. Edge sharpness bonus - reward sharp transitions at absorption window boundaries
            if len(wavelengths) > 10:  # Ensure enough points for gradient calculation
                absorption_gradient = np.abs(np.gradient(A))
                # Define edge regions around the absorption window boundaries
                edge_width = window_width * 0.2  # 20% of window width for edge detection
                lower_edge = (wavelengths >= (window_min - edge_width)) & (wavelengths <= (window_min + edge_width))
                upper_edge = (wavelengths >= (window_max - edge_width)) & (wavelengths <= (window_max + edge_width))
                edge_indices = np.where(lower_edge | upper_edge)[0]
                
                if len(edge_indices) > 0:
                    edge_sharpness = np.mean(absorption_gradient[edge_indices])
                    sharpness_bonus = penalty_weights['sharpness'] * np.clip(edge_sharpness, 0, 2)
                else:
                    sharpness_bonus = 0
            else:
                sharpness_bonus = 0
            
            # 8. Uniformity bonus within absorption window
            if np.sum(absorption_window) > 1:  # Need at least 2 points for std calculation
                absorption_uniformity = 1.0 - np.std(A[absorption_window])
                uniformity_bonus = penalty_weights['uniformity'] * np.clip(absorption_uniformity, 0, 1)
            else:
                uniformity_bonus = 0
            
            # 9. Additional penalty for broadband absorbers (high absorption everywhere)
            total_absorption = np.mean(A)
            if total_absorption > 0.8 and peak_absorption - off_peak_absorption < 0.2:
                # This is likely a broadband absorber, not selective
                broadband_penalty = -penalty_weights['anti_broadband'] * total_absorption
            else:
                broadband_penalty = 0
            
            # Combine all enhancement terms
            enhancement = (T_penalty + selectivity_bonus + peak_bonus + reflection_bonus + 
                         peak_T_penalty + off_peak_T_penalty + sharpness_bonus + 
                         uniformity_bonus + broadband_penalty)
        else:
            # Fallback if wavelength indexing fails
            enhancement = T_penalty
    else:
        # If no wavelength info, just penalize transmission
        enhancement = -penalty_weights['transmission'] * np.mean(T**2)
    
    # Final reward combines base matching with selective behavior enhancements
    final_reward = base_reward + enhancement_weight * enhancement
    
    return final_reward


def cal_reward_selective_adaptive(R, T, A, target, wavelengths=None, config=None):
    '''
    Adaptive wrapper for selective absorber rewards with preset configurations.
    
    This function provides an easy interface with common presets for different
    wavelength ranges and applications.
    
    Args:
        R, T, A: numpy arrays. Reflection, transmission, and absorption spectra
        target: dict. Target spectra {'R': array, 'T': array, 'A': array}
        wavelengths: numpy array. Wavelength points (in μm)
        config: dict or str. Configuration dictionary or preset name
                Supported presets: 'telecom_1550', 'nir_980', 'visible_633', 
                                 'nir_1500', 'custom'
        
    Returns:
        reward: float. Reward value for the given spectra
        
    Example:
        # Using preset
        reward = cal_reward_selective_adaptive(R, T, A, target, wavelengths, 'telecom_1550')
        
        # Using custom config
        custom_config = {
            'target_wavelength': 1.3,
            'window_width': 0.08,
            'enhancement_weight': 0.4,
            'penalty_weights': {'transmission': 20.0}
        }
        reward = cal_reward_selective_adaptive(R, T, A, target, wavelengths, custom_config)
    '''
    
    # Preset configurations for common applications
    presets = {
        'telecom_1550': {
            'target_wavelength': 1.55,
            'window_width': 0.05,  # ±50nm
            'enhancement_weight': 0.3,
            'penalty_weights': {'transmission': 15.0, 'selectivity': 5.0}
        },
        
        'nir_1500': {
            'target_wavelength': 1.5,
            'window_width': 0.05,  # ±50nm  
            'enhancement_weight': 0.3,
            'penalty_weights': {'transmission': 15.0, 'selectivity': 5.0}
        },
        
        'nir_980': {
            'target_wavelength': 0.98,
            'window_width': 0.04,  # ±40nm
            'enhancement_weight': 0.35,
            'penalty_weights': {'transmission': 18.0, 'selectivity': 6.0}
        },
        
        'visible_633': {
            'target_wavelength': 0.633,
            'window_width': 0.03,  # ±30nm
            'enhancement_weight': 0.4,
            'penalty_weights': {'transmission': 20.0, 'selectivity': 7.0, 'sharpness': 2.0}
        },
        
        'swir_2000': {
            'target_wavelength': 2.0,
            'window_width': 0.1,   # ±100nm
            'enhancement_weight': 0.25,
            'penalty_weights': {'transmission': 12.0, 'selectivity': 4.0}
        }
    }
    
    # Handle configuration input
    if config is None:
        config = 'nir_1500'  # Default to 1500nm
    
    if isinstance(config, str):
        if config in presets:
            config_dict = presets[config]
        else:
            raise ValueError(f"Unknown preset: {config}. Available: {list(presets.keys())}")
    else:
        config_dict = config
    
    # Call the configurable reward function with the specified configuration
    return cal_reward_selective_configurable(
        R, T, A, target, wavelengths,
        target_wavelength=config_dict.get('target_wavelength', 1.5),
        window_width=config_dict.get('window_width', 0.05),
        enhancement_weight=config_dict.get('enhancement_weight', 0.3),
        penalty_weights=config_dict.get('penalty_weights', None)
    )


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def batch_spectrum(env, names_list, thickness_list):

    def spectrum(args):
        '''
        Inputs: 
            1. names: list of lists, each list correspond to the structures
            2. thickness: list of lists
        '''
        names, thickness = args
        R, T, A = env.spectrum(names, thickness, 0, False)

        return R, T, A

    res = Parallel(n_jobs=num_workers)(delayed(spectrum)(args)
                                       for args in
                                       zip(names_list, thickness_list))
    res = np.array(res)
    Rs, Ts, As = res[:, 0, :], res[:, 1, :], res[:, 2, :]

    return Rs, Ts, As


def merge_layers(categories, thicknesses):
    '''
    Merges consecutive layers with the same material types.
    '''

    thicknesses = thicknesses[1:-1]
    c_output = [categories[0]]
    t_output = [thicknesses[0]]
    for i, (c, d) in enumerate(zip(categories[1:], thicknesses[1:])):

        if c == c_output[-1]:
            t_output[-1] += d
            continue
        else:
            c_output.append(c)
            t_output.append(d)

    t_output.insert(0, np.inf)
    t_output.insert(len(t_output), np.inf)

    return c_output, t_output


def get_structure(categories, values, materials, ds, continuous=False,
                  max_value=400):
    '''
    Given categories and values, return the strucure in the form 
    (name (str), thickness (nm))
    '''

    def threshold(value):
        '''

        '''

    names = [materials[item] for item in categories]

    if not continuous:
        thickness = [np.inf] + [ds[item] for item in values] + [np.inf]
    else:
        thickness = []
        for category, value in zip(categories, values):
            name = materials[category]
            if name == 'Ag':
                thickness.append(
                    min(max(15, int(value * max_value//2)), max_value))
            elif name in METALS:
                thickness.append(
                    min(max(5, int(value * max_value//2)), max_value))
            elif name in INSULATORS:
                thickness.append(
                    min(max(1, int(value * max_value//2)), max_value))
            else:
                raise ValueError('Material not known')
        # thickness = [np.inf] + [min(max(5, int(item * 2e2)), 200) for i,
        # item in enumerate(values)] + [np.inf]
        thickness = [np.inf] + thickness + [np.inf]
    return names, thickness

class DesignTracker():
    def __init__(self, epochs, **kwargs):
        """
        This class tracks the best designs discovered.
        """
        if epochs == -1:
            self.layer_ls = []
            self.thick_ls = []
            self.max_ret_ls = []
        self.layer_ls = [0] * epochs
        self.thick_ls = [0] * epochs
        self.max_ret_ls = [0] * epochs
        self.kwargs = kwargs
        self.current_e = 0

    def store(self, layers, thicknesses, ret, e, append_mode=False):
        
        if append_mode:
            self.layer_ls.append(layers)
            self.thick_ls.append(thicknesses)
            self.max_ret_ls.append(ret)

        else:
            if ret >= self.max_ret_ls[e]:
                self.layer_ls[e] = layers
                self.thick_ls[e] = thicknesses
                self.max_ret_ls[e] = ret

    def save_state(self):
        # save buffer from all processes
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        filename = os.path.join(self.kwargs['output_dir'], 'design_tracker_{}.pkl'.format(rank))
        pkl.dump(self, open(filename, 'wb'))
    
    def print_progress(self):
        progress = list(zip(self.layer_ls,  self.thick_ls, self.max_ret_ls))
        read_progress = []
        for i in range(len(progress)):
            if progress[i] == (0,0,0):
                break
            read_progress.append(['|'.join([l + ' ' + str(d) + ' nm' for l, d in zip(progress[i][0], progress[i][1])]) + ', Merit {:.3f}'.format(progress[i][2])])

        return read_progress

def print_progress(progress):

    for i in range(len(progress)):
        print(progress[i], 0)
        progress[i] = ['|'.join([l + ' ' + str(d) + ' nm' for l, d in zip(progress[i][0], progress[i][1])]), progress[i][2]]

    return progress

class TMM_sim():
    def __init__(self, mats=['Ge'], wavelength=np.arange(0.38, 0.805, 0.01), substrate='Cr', substrate_thick=500, data_path=None):
        '''
        This class returns the spectrum given the designed structures.
        '''
        self.mats = mats
        # include substrate
        self.all_mats = mats + [substrate] if substrate not in ['Glass', 'Air'] else mats
        self.wavelength = wavelength
        self.data_path = data_path if data_path is not None else DATABASE
        self.nk_dict = self.load_materials()
        self.substrate = substrate
        self.substrate_thick = substrate_thick

    def load_materials(self):
        '''
        Load material nk and return corresponding interpolators.

        Return:
            nk_dict: dict, key -- material name, value: n, k in the 
            self.wavelength range
        '''
        nk_dict = {}

        for mat in self.all_mats:
            nk = pd.read_csv(os.path.join(self.data_path, mat + '.csv'))
            nk.dropna(inplace=True)
            wl = nk['wl'].to_numpy()
            index = (nk['n'] + nk['k'] * 1.j).to_numpy()
            mat_nk_data = np.hstack((wl[:, np.newaxis], index[:, np.newaxis]))


            mat_nk_fn = interp1d(
                    mat_nk_data[:, 0].real, mat_nk_data[:, 1], kind='quadratic')
            nk_dict[mat] = mat_nk_fn(self.wavelength)

        return nk_dict

    # def spectrum(self, materials, thickness, theta=0, plot=False, title=False):
    #     '''
    #     Input:
    #         materials: list
    #         thickness: list
    #         theta: degree, the incidence angle

    #     Return:
    #         s: array, spectrum
    #     '''
    #     degree = pi/180
    #     if self.substrate != 'Air':
    #         thickness.insert(-1, self.substrate_thick) # substrate thickness

    #     R, T, A = [], [], []
    #     for i, lambda_vac in enumerate(self.wavelength * 1e3):

    #         # we assume the last layer is glass
    #         if self.substrate == 'Glass':
    #             n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1.45, 1]
    #         elif self.substrate == 'Air':
    #             n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1]
    #         else:
    #             n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict[self.substrate][i], 1]

    #         # n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict['Cr'][i]]

    #         # mport pdb; pdb.set_trace()
    #         res = coh_tmm('s', n_list, thickness, theta * degree, lambda_vac)
    #         R.append(res['R'])
    #         T.append(res['T'])

    #     R, T = np.array(R), np.array(T)
    #     A = 1 - R - T

    #     if plot:
    #         self.plot_spectrum(R, T, A)
    #         if title:
    #             thick = thickness[1:-1]
    #             title = ' | '.join(['{}nm {}'.format(d, m)
    #                                 for d, m in zip(thick, materials)])
    #             if self.substrate is not 'Air':
    #                 title = 'Air | ' + title + ' | {}nm {} '.format(self.substrate_thick, self.substrate) + '| Air'
    #             else:
    #                 title = 'Air | ' + title + ' | Air'
    #             plt.title(title, **{'size': '10'})

    #     return R, T, A


    # In utils.py inside the TMM_sim class

    # def spectrum(self, materials, thickness, theta=0, plot=False, title=False):
    #     print("Deneme")

    def spectrum(self, materials, thickness, theta=0, plot=False, title=False):
        '''
        Input:
            materials: list of layer material names (e.g., ['TiO2', 'SiO2'])
            thickness: list including boundary conditions (e.g., [np.inf, d1, d2, ..., dn, np.inf])
            theta: degree, the incidence angle
            plot: boolean, whether to plot the result
            title: boolean or string, title for the plot

        Return:
            R, T, A: numpy arrays for Reflection, Transmission, Absorption
        '''
        degree = pi / 180
        R, T, A = [], [], []

        # Extract only the actual layer thicknesses (d1, d2, ..., dn)
        # Assumes input 'thickness' is [inf, d1, ..., dn, inf]
        if len(thickness) < 2:
            raise ValueError("Input 'thickness' list must at least contain [np.inf, np.inf]")
        layer_thicknesses = thickness[1:-1]

        wavelengths_nm = self.wavelength * 1e3 # For tmm calculation

        for i, lambda_vac in enumerate(wavelengths_nm):

            # --- Construct n_list for tmm ---
            # Format: [n_ambient, n_layer1, ..., n_layerN, n_substrate]
            n_list_tmm = [1]  # Ambient medium (Air) n=1
            n_list_tmm.extend([self.nk_dict[mat][i] for mat in materials]) # Layers

            if self.substrate == 'Glass':
                n_list_tmm.append(1.45) # Substrate n
            elif self.substrate == 'Air':
                # No substrate layer, the final medium is Air (n=1)
                # It will be handled by the final element in d_list being inf
                n_list_tmm.append(1) # Append final ambient index
            elif self.substrate in self.nk_dict:
                n_list_tmm.append(self.nk_dict[self.substrate][i]) # Substrate n
            else:
                raise ValueError(f"Substrate material '{self.substrate}' n,k data not loaded.")


            # --- Construct d_list for tmm ---
            # Format: [inf, d_layer1, ..., d_layerN, d_substrate_or_inf]
            d_list_tmm = [np.inf] # Thickness of incident medium
            d_list_tmm.extend(layer_thicknesses) # Add actual layer thicknesses

            if self.substrate == 'Air':
                # Final medium is air, thickness is effectively infinite
                d_list_tmm.append(np.inf)
            else:
                # Use substrate thickness if defined, otherwise infinite (semi-infinite substrate)
                d_list_tmm.append(np.inf if self.substrate_thick == np.inf else self.substrate_thick)


            # --- Perform TMM calculation ---
            try:
                # Note: tmm library might interpret the last n in n_list as the final medium
                # if the last d in d_list is np.inf. Let's adjust n_list if necessary.
                # Standard usage: n_list = [n_i, n_1, ..., n_N, n_f], d_list = [inf, d_1, ..., d_N, inf]

                # Let's adapt to the standard documented usage:
                final_n_list = [1] # n_incident (Air)
                final_n_list.extend([self.nk_dict[mat][i] for mat in materials]) # n_layers
                final_d_list = [np.inf] # d_incident
                final_d_list.extend(layer_thicknesses) # d_layers

                # Add substrate index and final ambient index/thickness
                if self.substrate == 'Glass':
                    final_n_list.append(1.45) # n_substrate
                    # If substrate_thick is finite, tmm considers it a layer before final medium
                    if self.substrate_thick != np.inf:
                        final_d_list.append(self.substrate_thick) # d_substrate
                        final_n_list.append(1) # n_final (Air)
                        final_d_list.append(np.inf) # d_final
                    else: # Semi-infinite substrate
                        final_n_list.append(1) # n_final (Air) - tmm treats last n as final medium if last d=inf
                        final_d_list.append(np.inf) # d_substrate = inf

                elif self.substrate == 'Air':
                    # No substrate, final medium is Air
                    final_n_list.append(1) # n_final (Air)
                    final_d_list.append(np.inf) # d_final

                elif self.substrate in self.nk_dict:
                    final_n_list.append(self.nk_dict[self.substrate][i]) # n_substrate
                    if self.substrate_thick != np.inf:
                        final_d_list.append(self.substrate_thick) # d_substrate
                        final_n_list.append(1) # n_final (Air)
                        final_d_list.append(np.inf) # d_final
                    else: # Semi-infinite substrate
                        final_n_list.append(1) # n_final (Air)
                        final_d_list.append(np.inf) # d_substrate = inf
                else:
                    raise ValueError(f"Substrate material '{self.substrate}' n,k data not loaded.")


                # Ensure lengths match before calling coh_tmm
                if len(final_n_list) != len(final_d_list):
                    raise ValueError(f"Mismatch lengths! n_list({len(final_n_list)}) d_list({len(final_d_list)})")

                res = coh_tmm('s', final_n_list, final_d_list, theta * degree, lambda_vac)
                R.append(res['R'])
                T.append(res['T'])

            except ValueError as e:
                print(f"Error in coh_tmm at lambda={lambda_vac:.0f}nm, theta={theta:.1f}deg: {e}")
                # Optional: Print lists for debugging
                # print(f"  n_list ({len(final_n_list)}): {[f'{x:.2f}' for x in final_n_list]}")
                # print(f"  d_list ({len(final_d_list)}): {final_d_list}")
                R.append(np.nan) # Append NaN on error
                T.append(np.nan)
                # Or re-raise the error if you want execution to stop:
                # raise e

        R, T = np.array(R), np.array(T)
        # Calculate A, handle potential NaNs from errors
        A = 1 - R - T
        A[np.isnan(A)] = 0 # Or handle as needed, e.g., set to 0 or propagate NaN

        if plot:
            self.plot_spectrum(R, T, A) # Assuming plot_spectrum is another method in the class
            if title:
                plot_title = title if isinstance(title, str) else ' | '.join([f'{d}nm {m}' for d, m in zip(layer_thicknesses, materials)])
                # Add substrate info to title if needed
                plt.title(plot_title, **{'size': '10'})

        return R, T, A



    def plot_spectrum(self, R, T, A):

        plt.plot(self.wavelength * 1000, R, self.wavelength *
                 1000, T, self.wavelength * 1000, A, linewidth=3)
        plt.ylabel('R/T/A')
        plt.xlabel('Wavelength (nm)')
        plt.legend(['R: Average = {:.2f}%'.
                    format(np.mean(R)*100),
                    'T: Average = {:.2f}%'.
                    format(np.mean(T)*100),
                    'A: Average = {:.2f}%'.
                    format(np.mean(A)*100)])
        plt.grid('on', linestyle='--')
        plt.ylim([0, 1])


# Plotting utils
def visualize_progress(file, x, ax=None, color='b', alpha=1):
    df = pd.read_csv(file, sep="\t")
    width = 0.5
    # x = 'Time'
    if ax is None:
        fig, ax = plt.subplots(2,1)
    sns.lineplot(x=x, y='MaxEpRet', data=df, ax=ax[0], color=color, alpha=alpha)
    # ax[0].legend(['Max {}'.format(np.max(df['MaxEpRet']))])
    sns.lineplot(x=x, y='AverageEpRet', data=df,
                 ax=ax[1], color=color, alpha=alpha)
    plt.fill_between(df[x],
                     df['AverageEpRet']-width/2*df['StdEpRet'],
                     df['AverageEpRet']+width/2*df['StdEpRet'],
                     alpha=0.3, color=color)

    return df

def combine_tracker(folder):
    '''
    Merge all buffers
    '''
    trackers = []
    
    if 'design_tracker_merged.pkl' in os.listdir(folder):
        tracker_file = os.path.join(folder, 'design_tracker_merged.pkl')
        combined_tracker = pkl.load(open(tracker_file, 'rb'))
        return combined_tracker

    for file in os.listdir(folder):
        if file.startswith('design_tracker_'):
            tracker_file = os.path.join(folder, file)
            trackers.append(pkl.load(open(tracker_file, 'rb')))        

    combined_tracker = DesignTracker(len(trackers[0].layer_ls))
    max_idx = np.argmax(np.array([tracker.max_ret_ls for tracker in trackers]), axis=0)
    for e in range(len(trackers[0].layer_ls)):
        combined_tracker.layer_ls[e] = trackers[max_idx[e]].layer_ls[e]
        combined_tracker.thick_ls[e] = trackers[max_idx[e]].thick_ls[e]
        combined_tracker.max_ret_ls[e] = trackers[max_idx[e]].max_ret_ls[e]
    
    if combined_tracker.layer_ls[-1] != 0:
        tracker_file = os.path.join(folder, 'design_tracker_merged.pkl')
        pkl.dump(combined_tracker, open(os.path.join(folder, tracker_file), 'wb'))

    return combined_tracker

def summarize_res(exp_ls, seed_ls, color, alpha, x='Epoch'):
        
    root = '../spinningup/data/'
    progress_ls = []
    max_ret_ls = []

    params = {'size':14}
    matplotlib.rc('font', **params)

    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for a, c, exp, seed in zip(alpha, color, exp_ls, seed_ls):
        folder = os.path.join(root, exp, exp+'_s{}'.format(seed))
        progress_file = os.path.join(folder, 'progress.txt')
        df = visualize_progress(progress_file, x=x, ax=ax, color=c, alpha=a)

        tracker = combine_tracker(folder)
        progress = tracker.print_progress()
        print('{}, Best discovered so far {}'.format(exp, progress[np.argmax(tracker.max_ret_ls)]))
        progress_ls.append(progress)
        max_ret_ls.append('Max merit {:.3f}'.format(np.max(df['MaxEpRet'])))

    ax[0].legend(max_ret_ls)
    ax[1].legend(exp_ls)
    plt.show()
    return progress_ls

def load_exp_res(folder):
    subfolders = [item for item in glob.glob(folder+'/*')]

    def read_hyper(file_name, rep=10):

        with open(os.path.join(file_name, 'config.json')) as f:
            hypers = json.load(f)
            hypers_dict = {}
            for k, v in hypers.items():
                if k.startswith('logger'):
                    continue
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, list):
                            hypers_dict[str(k)+'_'+str(kk)] = [vv[0]]*rep
                        else:
                            hypers_dict[str(k)+'_'+str(kk)] = [vv]*rep
                else: 
                    hypers_dict[k] = [v] * rep
            
            hyper_df = pd.DataFrame(hypers_dict)
            return hyper_df 

    first=True # first pandas file to load
    for subfolder in tqdm(subfolders):
        runs = glob.glob(subfolder+'/*')
        num_epochs = len(pd.read_csv(os.path.join(runs[0], 'progress.txt'),sep='\t'))
        for run in runs:

            tracker = combine_tracker(run)
            progress = tracker.print_progress()
            best_design = progress[np.argmax(tracker.max_ret_ls)]

            if first:
                df = pd.read_csv(os.path.join(run, 'progress.txt'),sep='\t')
                hyper_df = read_hyper(run, rep=len(df))
                best_designs_df = pd.DataFrame([{'best_design':best_design}]*len(df))
                df = pd.concat([df, hyper_df, best_designs_df], axis=1)
                first = False

            else:
                df_ = pd.read_csv(os.path.join(run, 'progress.txt'),sep='\t')
                hyper_df = read_hyper(run, rep=len(df_))
                best_designs_df = pd.DataFrame([{'best_design':best_design}]*len(df_))
                df_ = pd.concat([df_, hyper_df, best_designs_df], axis=1)
                df = pd.concat([df, df_], axis=0)   

    return df   


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
        bounds = [(15, 200)] * len(x0)
    
    res = minimize(objective_func, x0, bounds=bounds, options={'disp':True})
    x_opt = [int(item) for item in res.x]
    
    if display:
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x0+[np.inf], title=True, plot=True)
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x_opt+[np.inf], title=True, plot=True)
    
    return x_opt, res


import numpy as np # Ensure numpy is imported

def cal_reward_lorentzian_absorption(R, T, A, wavelengths,
                                     absorption_center, absorption_hwhm,
                                     lorentzian_absorption_target,
                                     reflection_target,
                                     transmission_target):
    """
    Calculates the reward for a structure aiming for a Lorentzian absorption peak.

    Args:
        R (np.ndarray): Simulated reflection spectrum.
        T (np.ndarray): Simulated transmission spectrum.
        A (np.ndarray): Simulated absorption spectrum.
        wavelengths (np.ndarray): Wavelength points (in µm).
        absorption_center (float): Center of the target Lorentzian peak (in µm).
        absorption_hwhm (float): Half-width at half-maximum of the target peak (in µm).
        lorentzian_absorption_target (np.ndarray): The ideal Lorentzian absorption spectrum.
        reflection_target (np.ndarray): The ideal reflection spectrum (1 - lorentzian_absorption_target).
        transmission_target (np.ndarray): The ideal transmission spectrum (all zeros).

    Returns:
        float: The calculated reward.
    """

    # 1. Transmission Penalty (Crucial)
    # Strong penalty for any transmission, squared to penalize larger values more.
    transmission_penalty_weight = 20.0  # Tunable
    # Ensure T is always non-negative before squaring, though it should be.
    # Using np.mean(np.maximum(T, 0)**2) for robustness, though T should be >= 0.
    reward_transmission = -transmission_penalty_weight * np.mean(T**2)


    # 2. Lorentzian Absorption Peak Fitness (Primary Goal for Absorption)
    # Focus on the region around the peak, e.g., center +/- 4*HWHM, or full range if target is defined that way
    # The `lorentzian_absorption_target` should ideally be zero outside the desired peak region.

    # We will calculate MSE against the provided lorentzian_absorption_target directly.
    # This target should be correctly shaped (Lorentzian in band, zero outside).
    mse_absorption_overall = np.mean((A - lorentzian_absorption_target)**2)

    absorption_fitness_weight = 1.5 # Tunable, increased weight as it's a primary objective
    # Reward for matching the Lorentzian shape (1 - MSE)
    # MSE is always >= 0. So 1-MSE can be <=1.
    reward_absorption_shape = absorption_fitness_weight * (1.0 - mse_absorption_overall)


    # 3. Reflection Fitness
    # The reflection_target is `1 - lorentzian_absorption_target` (assuming T_target is zero)
    mse_reflection_overall = np.mean((R - reflection_target)**2)
    reflection_fitness_weight = 1.0 # Tunable
    reward_reflection_shape = reflection_fitness_weight * (1.0 - mse_reflection_overall)

    # Total Reward:
    # Start with a base that can go negative (e.g. -MSEs) and add positive components,
    # or scale 1-MSE terms.
    # For now, sum of weighted (1-MSE) terms and direct penalties.
    # The reward_transmission is already a penalty.

    # Normalizing the MSE components to be rewards:
    # Max reward for absorption shape is absorption_fitness_weight
    # Max reward for reflection shape is reflection_fitness_weight
    # Max penalty for transmission is -infinity, but practically bounded by -transmission_penalty_weight if T=1 everywhere.

    total_reward = (reward_transmission +
                    reward_absorption_shape +
                    reward_reflection_shape
                   )

    # Adding a small constant bonus if peak absorption is high and in the right place
    # to encourage finding the peak initially.
    peak_idx = np.argmin(np.abs(wavelengths - absorption_center))
    if A[peak_idx] > 0.8 and lorentzian_absorption_target[peak_idx] > 0.8 : # if actual peak is high and target peak is high
        # and if the max absorption is close to the center
        if np.abs(wavelengths[np.argmax(A)] - absorption_center) < absorption_hwhm :
             total_reward += 0.5 # Small encouragement bonus

    return total_reward

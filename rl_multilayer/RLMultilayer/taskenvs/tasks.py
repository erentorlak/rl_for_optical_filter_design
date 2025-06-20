"""
Return tasks
"""
import numpy as np
from RLMultilayer.taskenvs.task_envs import TMM
from RLMultilayer.utils import TMM_sim, cal_reward, cal_reward_selective_1500, cal_reward_selective_configurable, cal_reward_selective_adaptive, cal_reward_lorentzian_absorption # Added new reward function
import gym

eps=1e-5


def get_env_fn(env_name, **kwargs):

    if env_name == 'PerfectAbsorberVis-v0':
        return perfect_absorber_vis(**kwargs)
    elif env_name == 'PerfectAbsorberVisNIR-v0':
        return perfect_absorber_vis_nir(**kwargs)
    elif env_name == 'PerfectAbsorberVisNIR-v1':
        return perfect_absorber_vis_nir5(**kwargs)
    elif env_name == 'BandpassFilter-v0':
        return bandpass_filter_task(**kwargs)
    elif env_name == 'absorber_filter_task':
        return absorber_filter_task(**kwargs)
    elif env_name == 'erent_1500_absorber_task_v0':
        return erent_1500_absorber_task_v0(**kwargs)
    elif env_name == 'LorentzianAbsorber2000nm-v0': # New condition
        return lorentzian_absorption_task(**kwargs) # Calling the new task function
    else:
        try:
            return lambda: gym.make(env_name)
        except:
            raise NotImplementedError("Env not registered!")

#####################################
# Perfect absorber in visible range #
#####################################


def perfect_absorber_vis(**kwargs):

    lamda_low = 0.4
    lamda_high = 0.8
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.02)
    materials = ['Ag', 'Al', 'Cr', 'Ge', 'SiC',
                 'HfO2', 'SiO2', 'Al2O3', 'MgF2', 'TiO2']
    simulator = TMM_sim(materials, wavelengths)
    thickness_list = np.arange(15, 201, 2)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make

#####################################
# Perfect absorber in [0.4-2] um #
#####################################

def perfect_absorber_vis_nir(**kwargs):

    def gen_grid(low, high):
    
        ws = [low]
        while ws[-1] < high:
            ws.append(1.03*ws[-1])

        return np.array(ws)

    lamda_low = 0.4
    lamda_high = 2.0
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    materials = ['Ag', 'Al', 'Al2O3', 'Cr', 'Ge', 'HfO2', 'MgF2', 'Ni', 'Si', 'SiO2', 'Ti', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Fe2O3']
    simulator = TMM_sim(materials, wavelengths, substrate='Glass', substrate_thick=500)
    thickness_list = np.arange(15, 201, 5)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              **kwargs}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make


def perfect_absorber_vis_nir5(**kwargs):

    def gen_grid(low, high):
    
        ws = [low]
        while ws[-1] < high:
            ws.append(1.03*ws[-1])

        return np.array(ws)

    lamda_low = 0.4
    lamda_high = 2.0
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    materials = ['Cr', 'Ge', 'Si', 'TiO2', 'MgF2']
    simulator = TMM_sim(materials, wavelengths, substrate='Glass', substrate_thick=500)
    thickness_list = np.arange(15, 201, 5)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              **kwargs}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make



# this function only defined for 1800nm - 2200 nm. It does not care other wavelenghts.
def bandpass_filter_task(**kwargs):

    # --- 1. Dalga boyu aralığı tanımlanıyor ---
    lamda_low = 1.8  # 1800 nm
    lamda_high = 2.2 # 2200 nm
    wavelengths = np.arange(lamda_low, lamda_high + 1e-3, 0.5)  # 0.5 nm çözünürlük

    # --- 2. Malzemeler belirleniyor ---
    materials = ['SiO2', 'TiO2', 'Ge', 'Si', 'Al2O3', 'MgF2']  
    # (Kodda daha fazla malzeme varsa ekleyebilirsin)

    simulator = TMM_sim(materials, wavelengths, substrate='Glass', substrate_thick=500)

    # --- 3. Kalınlık listesi tanımlanıyor ---
    thickness_list = np.arange(10, 1001, 10)
    # 10 nm - 1000 nm arası, 10 nm adım

    # --- 4. Hedef reflektans spektrumu tanımlanıyor ---
    w0 = 2.0   # 2000 nm merkez
    gama = 0.05  # 50 nm genişlik = 0.05 um
    L = 1 - gama**2 / ((wavelengths - w0)**2 + gama**2)  # Reflection hedefi (L -> 1)

    target = {'R': L}

    # --- 5. Ortam config ayarlanıyor ---
    config = {
        'wavelengths': wavelengths,
        'materials': materials,
        'target': target,
        'merit_func': cal_reward,
        'simulator': simulator,
        **kwargs
    }

    if kwargs.get('discrete_thick', False):
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        return TMM(**config)

    return make





def absorber_filter_task(**kwargs):

    # --- 1. Define wavelength range: 1000–3000 nm ---
    lamda_low = 1.5   # 1500 nm
    lamda_high = 2.5  # 2500 nm
    wavelengths = np.arange(lamda_low, lamda_high + 1e-3, 0.01)  # 10 nm resolution

    # --- 2. Define candidate materials: metals + dielectrics ---
    materials = ['TiO2', 'SiO2', 'Ge', 'Si', 'Ni', 'Cr', 'Al2O3']

    # --- 3. Use a metal substrate to block transmission ---
    substrate_material = 'Ag'            # strongly reflective, no T
    substrate_thickness = 500        # semi-infinite metal layer

    # --- 4. Create simulator ---
    simulator = TMM_sim(
        mats=materials,
        wavelength=wavelengths,
        substrate=substrate_material,
        substrate_thick=substrate_thickness
    )

    # --- 5. Allowed thickness values ---
    thickness_list = np.arange(10, 1001, 10)  # 10–1000 nm

    # --- 6. Define full-band targets ---
    w0 = 2.0     # center: 2000 nm
    gamma = 0.05 # width = 50 nm

    # Full default: reflect everything, absorb nothing, transmit nothing
    R_target = np.ones_like(wavelengths)
    T_target = np.zeros_like(wavelengths)  # <-- force T = 0 across ALL wavelengths

    # In absorption band: suppress reflection → increase absorption
    absorption_band = (wavelengths >= 1.8) & (wavelengths <= 2.2)
    R_target[absorption_band] = 1 - gamma**2 / ((wavelengths[absorption_band] - w0)**2 + gamma**2)

    # --- 7. Define target dictionary ---
    target = {'R': R_target, 'T': T_target}

    # --- 8. Build environment config ---
    config = {
        'wavelengths': wavelengths,
        'materials': materials,
        'target': target,
        'merit_func': cal_reward,
        'simulator': simulator,
        **kwargs
    }

    if kwargs.get('discrete_thick', False):
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        return TMM(**config)

    return make


def erent_1500_absorber_task_v0(**kwargs):
    """
    Defines a task for designing a selective absorber with perfect absorption at 1500nm.
    The agent's goal is to generate a structure with a spectrum that:
    1. Has perfect absorption (A=1) in the 1450-1550nm range
    2. Has perfect reflection (R=1) at all other wavelengths  
    3. Has zero transmission (T=0) across the entire spectrum
    
    This is ideal for applications requiring selective energy absorption at 1500nm
    while reflecting all other wavelengths for maximum efficiency.
    """
    
    # 1. Define the wavelength range (in micrometers)
    # Extended range to capture both visible and near-infrared
    lamda_low = 0.4   # 400 nm
    lamda_high = 2.0  # 2000 nm 
    wavelengths = np.arange(lamda_low, lamda_high + 1e-3, 0.01)  # 10 nm resolution
    
    # 2. Define the set of available materials for the agent to choose from
    # Use materials that have good wavelength coverage for 1500nm (1.5 μm)
    # Based on the wavelength analysis: Cr (0.29-2.95), Ge (0.19-5.0), TiO2 (0.3-15.0), etc.
    materials = ['Cr', 'Ge', 'TiO2', 'SiO2', 'Al2O3', 'MgF2']
    
    # 3. Initialize the TMM simulator 
    # Use metallic substrate to prevent transmission
    simulator = TMM_sim(
        mats=materials, 
        wavelength=wavelengths, 
        substrate='Cr',  # Metallic substrate to block transmission
        substrate_thick=500
        # Uses default data path './data' which contains all materials
    )
    
    # 4. Define the discrete thickness values the agent can select (in nm)
    thickness_list = np.arange(15, 251, 5)  # 15 nm to 250 nm, in 5 nm steps
    
    # 5. Define the Target Spectrum for selective absorption at 1500nm
    
    # Define absorption window: 1450-1550nm (1.45-1.55 μm)
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    # Target absorption: A=1 in absorption window, A=0 elsewhere
    target_A = np.zeros_like(wavelengths)
    target_A[absorption_window] = 1.0
    
    # Target reflection: R=0 in absorption window, R=1 elsewhere
    target_R = np.ones_like(wavelengths)
    target_R[absorption_window] = 0.0
    
    # Target transmission: T=0 everywhere (no transmission desired)
    target_T = np.zeros_like(wavelengths)
    
    # Bundle the targets into a dictionary
    target = {'A': target_A, 'R': target_R, 'T': target_T}
    
    # 6. Configure the environment
    config = {
        'wavelengths': wavelengths,
        'materials': materials,
        'target': target,
        'merit_func': lambda R, T, A, target: cal_reward_selective_1500(
            R, T, A, target, wavelengths, target_wavelength=1.5
        ),
        'simulator': simulator,
        **kwargs
    }
    
    # Handle the discrete thickness setting
    if kwargs.get('discrete_thick', False):
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list
    
    # Return a function that creates the environment instance
    def make():
        env = TMM(**config)
        return env
    
    return make


############################################################
# Lorentzian Absorption Peak Task (1975-2025 nm)           #
############################################################
def lorentzian_absorption_task(**kwargs):

    # 1. Define Wavelengths (0.4 to 2.5 µm)
    lamda_low = 0.4
    lamda_high = 2.5
    # Ensure consistent high resolution, e.g., 5nm or 10nm. Using 10nm for broader range.
    wavelengths = np.arange(lamda_low, lamda_high + 1e-3, 0.01)

    # 2. Define Materials
    # Comprehensive list covering VIS to NIR, including good metals and dielectrics
    materials = ['Ag', 'Al', 'Al2O3', 'Cr', 'Ge', 'HfO2', 'MgF2', 'Ni', 'Si', 'SiO2', 'Ti', 'TiO2', 'Fe2O3', 'SiC', 'Si3N4', 'TiN', 'ZnO', 'ZnS', 'ZnSe']
    # Ensure these materials have .csv files in the 'data/' directory and cover the spectrum

    # 3. Instantiate Simulator (TMM_sim)
    # Using 'Ag' (Silver) as substrate to ensure high reflection and no transmission.
    # Thickness of 200nm for substrate is optically opaque for Ag.
    simulator = TMM_sim(materials, wavelengths, substrate='Ag', substrate_thick=200)

    # 4. Define Thickness List (for discrete optimization)
    # 10 nm to 300 nm in 5 nm steps
    thickness_list = np.arange(10, 301, 5)

    # 5. Define Target Spectrum Parameters for Lorentzian Peak
    absorption_center = 2.0  # 2000 nm
    absorption_fwhm = 0.05   # 50 nm FWHM
    absorption_hwhm = absorption_fwhm / 2.0 # 25 nm HWHM

    # 6. Generate Ideal Target Spectra
    # Target Transmission: Zero everywhere
    target_T_ideal = np.zeros_like(wavelengths)

    # Target Lorentzian Absorption:
    # Peak is defined by Lorentzian shape, zero absorption outside a defined band
    # For the target array, let's make it zero outside a slightly wider band than FWHM, e.g., +/- 10*HWHM
    lorentzian_band_min = absorption_center - 10 * absorption_hwhm # 2.0 - 0.25 = 1.75 µm
    lorentzian_band_max = absorption_center + 10 * absorption_hwhm # 2.0 + 0.25 = 2.25 µm

    lorentzian_A_ideal = np.zeros_like(wavelengths)
    active_indices = (wavelengths >= lorentzian_band_min) & (wavelengths <= lorentzian_band_max)

    lorentzian_A_ideal[active_indices] = (absorption_hwhm**2) / \
                                     ((wavelengths[active_indices] - absorption_center)**2 + absorption_hwhm**2)

    # Target Reflection: 1 - Ideal Absorption (since T_ideal is 0)
    target_R_ideal = 1.0 - lorentzian_A_ideal

    # 7. Prepare `target_for_env` dictionary (can be empty if lambda captures all)
    # This dict is passed to the merit function by the environment.
    # Our new reward function gets most params directly via lambda.
    target_for_env = {
        # Optional: could pass descriptive strings or the arrays themselves if preferred
        # 'description': 'Target for Lorentzian absorber 2000nm, 50nm FWHM'
    }

    # 8. Define Merit Function using a lambda to pass necessary parameters
    merit_func = lambda R_sim, T_sim, A_sim, target_dict_ignored: \
        cal_reward_lorentzian_absorption(
            R_sim, T_sim, A_sim,
            wavelengths,
            absorption_center, absorption_hwhm,
            lorentzian_A_ideal, # The ideal Lorentzian absorption shape
            target_R_ideal,     # The ideal reflection (1 - A_lorentzian)
            target_T_ideal      # The ideal transmission (all zeros)
        )

    # 9. Configure Environment
    config = {
        'wavelengths': wavelengths,
        "materials": materials,
        'target': target_for_env,
        "merit_func": merit_func,
        "simulator": simulator,
        **kwargs  # Pass through other kwargs like 'discrete_thick', 'max_layers'
    }

    if kwargs.get('discrete_thick', True): # Default to discrete thickness
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list
    else:
        config['discrete_thick'] = False


    # 10. Return a make function that creates the environment instance
    def make():
        env = TMM(**config)
        return env

    return make
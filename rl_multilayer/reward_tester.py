"""
Reward Function Tester for Selective Absorber at 1500nm

This module tests the cal_reward_selective_1500 function with various edge cases
and scenarios to ensure robust behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from RLMultilayer.utils import cal_reward_selective_1500


def test_perfect_case():
    """Test with perfect selective absorber case"""
    print("Testing perfect selective absorber case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)  # 400-2100nm in micrometers
    
    # Perfect selective absorber: A=1 in 1450-1550nm, R=1 elsewhere, T=0 everywhere
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    A_perfect = np.zeros_like(wavelengths)
    A_perfect[absorption_window] = 1.0
    
    R_perfect = np.ones_like(wavelengths)
    R_perfect[absorption_window] = 0.0
    
    T_perfect = np.zeros_like(wavelengths)
    
    target = {'A': A_perfect, 'R': R_perfect, 'T': T_perfect}
    
    reward = cal_reward_selective_1500(R_perfect, T_perfect, A_perfect, target, wavelengths)
    print(f"Perfect case reward: {reward:.4f}")
    assert reward > 0.9, f"Perfect case should have high reward, got {reward}"
    return reward


def test_worst_case():
    """Test with worst case scenario"""
    print("Testing worst case scenario...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Worst case: T=1 everywhere (full transmission, no absorption or reflection)
    A_worst = np.zeros_like(wavelengths)
    R_worst = np.zeros_like(wavelengths) 
    T_worst = np.ones_like(wavelengths)
    
    # Target still expects selective behavior
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_worst, T_worst, A_worst, target, wavelengths)
    print(f"Worst case reward: {reward:.4f}")
    assert reward < 0, f"Worst case should have negative reward, got {reward}"
    return reward


def test_broadband_absorber():
    """Test with broadband absorber (no selectivity)"""
    print("Testing broadband absorber case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Broadband absorber: A=1 everywhere, R=0 everywhere, T=0 everywhere
    A_broad = np.ones_like(wavelengths)
    R_broad = np.zeros_like(wavelengths)
    T_broad = np.zeros_like(wavelengths)
    
    # Target expects selectivity
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_broad, T_broad, A_broad, target, wavelengths)
    print(f"Broadband absorber reward: {reward:.4f}")
    # Should be lower than perfect case due to lack of selectivity
    return reward


def test_partial_transmission():
    """Test with partial transmission case"""
    print("Testing partial transmission case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Partial transmission: some T, some A, some R
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    A_partial = np.zeros_like(wavelengths)
    A_partial[absorption_window] = 0.7  # 70% absorption in target window
    
    R_partial = np.ones_like(wavelengths) * 0.8  # 80% reflection elsewhere
    R_partial[absorption_window] = 0.1  # 10% reflection in absorption window
    
    T_partial = np.ones_like(wavelengths) * 0.2  # 20% transmission everywhere (bad!)
    T_partial[absorption_window] = 0.2
    
    # Target expects no transmission
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_partial, T_partial, A_partial, target, wavelengths)
    print(f"Partial transmission reward: {reward:.4f}")
    # Should be penalized due to transmission
    return reward


def test_edge_sharpness():
    """Test edge sharpness detection"""
    print("Testing edge sharpness...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Create sharp-edged absorption peak
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    A_sharp = np.zeros_like(wavelengths)
    A_sharp[absorption_window] = 1.0
    
    R_sharp = np.ones_like(wavelengths)
    R_sharp[absorption_window] = 0.0
    
    T_sharp = np.zeros_like(wavelengths)
    
    target = {'A': A_sharp, 'R': R_sharp, 'T': T_sharp}
    
    reward_sharp = cal_reward_selective_1500(R_sharp, T_sharp, A_sharp, target, wavelengths)
    
    # Create gradual-edged absorption peak
    A_gradual = np.zeros_like(wavelengths)
    transition_indices = (wavelengths >= 1.44) & (wavelengths <= 1.56)
    for i, wl in enumerate(wavelengths):
        if 1.44 <= wl <= 1.45:
            A_gradual[i] = (wl - 1.44) / 0.01  # Gradual rise
        elif 1.45 < wl < 1.55:
            A_gradual[i] = 1.0
        elif 1.55 <= wl <= 1.56:
            A_gradual[i] = 1.0 - (wl - 1.55) / 0.01  # Gradual fall
    
    R_gradual = 1.0 - A_gradual
    T_gradual = np.zeros_like(wavelengths)
    
    reward_gradual = cal_reward_selective_1500(R_gradual, T_gradual, A_gradual, target, wavelengths)
    
    print(f"Sharp edge reward: {reward_sharp:.4f}")
    print(f"Gradual edge reward: {reward_gradual:.4f}")
    
    return reward_sharp, reward_gradual


def test_no_wavelength_info():
    """Test behavior when wavelength info is not provided"""
    print("Testing case with no wavelength information...")
    
    # Simple case with some absorption, reflection, transmission
    A_test = np.array([0.5, 0.6, 0.7])
    R_test = np.array([0.3, 0.2, 0.1])
    T_test = np.array([0.2, 0.2, 0.2])
    
    target = {'A': np.array([1.0, 1.0, 1.0]), 'R': np.array([0.0, 0.0, 0.0]), 'T': np.array([0.0, 0.0, 0.0])}
    
    reward = cal_reward_selective_1500(R_test, T_test, A_test, target, wavelengths=None)
    print(f"No wavelength info reward: {reward:.4f}")
    return reward


def test_broadband_reflector():
    """Test with broadband reflector case (should be penalized for no absorption)"""
    print("Testing broadband reflector case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Broadband reflector: R=1 everywhere, A=0 everywhere, T=0 everywhere
    A_reflector = np.zeros_like(wavelengths)
    R_reflector = np.ones_like(wavelengths)
    T_reflector = np.zeros_like(wavelengths)
    
    # Target expects selective absorption
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_reflector, T_reflector, A_reflector, target, wavelengths)
    print(f"Broadband reflector reward: {reward:.4f}")
    # Should be significantly penalized for no absorption in target window
    return reward


def test_narrow_absorption_peak():
    """Test with very narrow absorption peak (less selective than target)"""
    print("Testing narrow absorption peak...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Narrow peak: only 1490-1510nm absorption (narrower than 1450-1550nm target)
    narrow_window = (wavelengths >= 1.49) & (wavelengths <= 1.51)
    
    A_narrow = np.zeros_like(wavelengths)
    A_narrow[narrow_window] = 1.0
    
    R_narrow = np.ones_like(wavelengths)
    R_narrow[narrow_window] = 0.0
    
    T_narrow = np.zeros_like(wavelengths)
    
    # Target expects broader absorption
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_narrow, T_narrow, A_narrow, target, wavelengths)
    print(f"Narrow absorption peak reward: {reward:.4f}")
    return reward


def test_shifted_absorption_peak():
    """Test with absorption peak shifted away from target"""
    print("Testing shifted absorption peak...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Shifted peak: 1600-1700nm instead of 1450-1550nm
    shifted_window = (wavelengths >= 1.6) & (wavelengths <= 1.7)
    
    A_shifted = np.zeros_like(wavelengths)
    A_shifted[shifted_window] = 1.0
    
    R_shifted = np.ones_like(wavelengths)
    R_shifted[shifted_window] = 0.0
    
    T_shifted = np.zeros_like(wavelengths)
    
    # Target expects absorption at 1450-1550nm
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_shifted, T_shifted, A_shifted, target, wavelengths)
    print(f"Shifted absorption peak reward: {reward:.4f}")
    return reward


def test_weak_absorption():
    """Test with weak absorption in target window"""
    print("Testing weak absorption case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Weak absorption: only 30% absorption in target window
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    A_weak = np.zeros_like(wavelengths)
    A_weak[absorption_window] = 0.3  # Only 30% absorption
    
    R_weak = np.ones_like(wavelengths)
    R_weak[absorption_window] = 0.7  # 70% still reflected in target window
    
    T_weak = np.zeros_like(wavelengths)
    
    # Target expects perfect absorption
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_weak, T_weak, A_weak, target, wavelengths)
    print(f"Weak absorption reward: {reward:.4f}")
    return reward


def test_multilayer_interference():
    """Test realistic multilayer interference pattern"""
    print("Testing multilayer interference pattern...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Simulate realistic multilayer interference with oscillations
    np.random.seed(123)  # Different seed for different pattern
    
    # Create interference pattern with primary peak at 1500nm
    primary_peak = np.exp(-((wavelengths - 1.5)**2) / (2 * 0.05**2))  # Gaussian at 1500nm
    
    # Add secondary oscillations (multilayer interference)
    oscillations = 0.1 * np.sin(2 * np.pi * wavelengths / 0.1) * np.exp(-((wavelengths - 1.5)**2) / (2 * 0.3**2))
    
    A_interference = 0.05 + 0.8 * primary_peak + oscillations
    A_interference = np.clip(A_interference, 0, 1)
    
    # Complementary reflection with some baseline
    R_interference = 0.85 - 0.7 * primary_peak + 0.1 * np.random.normal(0, 0.02, len(wavelengths))
    R_interference = np.clip(R_interference, 0, 1)
    
    # Small transmission due to imperfect blocking
    T_interference = 0.02 + 0.03 * np.random.normal(0, 0.01, len(wavelengths))
    T_interference = np.clip(T_interference, 0, 1)
    
    # Ensure energy conservation
    total = A_interference + R_interference + T_interference
    A_interference /= total
    R_interference /= total
    T_interference /= total
    
    # Target
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_interference, T_interference, A_interference, target, wavelengths)
    print(f"Multilayer interference reward: {reward:.4f}")
    return A_interference, R_interference, T_interference, reward


def test_realistic_case():
    """Test a realistic multilayer structure case"""
    print("Testing realistic multilayer case...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    
    # Simulate a realistic response with some noise and imperfections
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    
    # Add some realistic noise and imperfections
    np.random.seed(42)  # For reproducible results
    
    A_realistic = np.random.normal(0.05, 0.02, len(wavelengths))  # Base low absorption with noise
    A_realistic[absorption_window] = np.random.normal(0.85, 0.05, np.sum(absorption_window))  # High absorption in target
    A_realistic = np.clip(A_realistic, 0, 1)
    
    R_realistic = np.random.normal(0.85, 0.05, len(wavelengths))  # High reflection baseline
    R_realistic[absorption_window] = np.random.normal(0.1, 0.05, np.sum(absorption_window))  # Low reflection in target
    R_realistic = np.clip(R_realistic, 0, 1)
    
    T_realistic = np.random.normal(0.05, 0.02, len(wavelengths))  # Low transmission with noise
    T_realistic = np.clip(T_realistic, 0, 1)
    
    # Ensure energy conservation (approximately)
    total = A_realistic + R_realistic + T_realistic
    A_realistic /= total
    R_realistic /= total
    T_realistic /= total
    
    # Target
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    reward = cal_reward_selective_1500(R_realistic, T_realistic, A_realistic, target, wavelengths)
    print(f"Realistic case reward: {reward:.4f}")
    return reward


def visualize_test_cases():
    """Visualize different test cases with reward values"""
    print("Creating visualizations...")
    
    wavelengths = np.arange(0.4, 2.1, 0.01)
    wavelengths_nm = wavelengths * 1000  # Convert to nm for plotting
    
    # Create target for all cases
    absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
    A_target = np.zeros_like(wavelengths)
    A_target[absorption_window] = 1.0
    R_target = np.ones_like(wavelengths)
    R_target[absorption_window] = 0.0
    T_target = np.zeros_like(wavelengths)
    target = {'A': A_target, 'R': R_target, 'T': T_target}
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Perfect case
    A_perfect = np.zeros_like(wavelengths)
    A_perfect[absorption_window] = 1.0
    R_perfect = np.ones_like(wavelengths)
    R_perfect[absorption_window] = 0.0
    T_perfect = np.zeros_like(wavelengths)
    reward_perfect = cal_reward_selective_1500(R_perfect, T_perfect, A_perfect, target, wavelengths)
    
    axes[0,0].plot(wavelengths_nm, A_perfect, 'r-', label='Absorption', linewidth=2)
    axes[0,0].plot(wavelengths_nm, R_perfect, 'b-', label='Reflection', linewidth=2)
    axes[0,0].plot(wavelengths_nm, T_perfect, 'g-', label='Transmission', linewidth=2)
    axes[0,0].set_title(f'Perfect Selective Absorber\nReward: {reward_perfect:.3f}')
    axes[0,0].set_xlabel('Wavelength (nm)')
    axes[0,0].set_ylabel('R/T/A')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Broadband absorber
    A_broad = np.ones_like(wavelengths)
    R_broad = np.zeros_like(wavelengths)
    T_broad = np.zeros_like(wavelengths)
    reward_broad = cal_reward_selective_1500(R_broad, T_broad, A_broad, target, wavelengths)
    
    axes[0,1].plot(wavelengths_nm, A_broad, 'r-', label='Absorption', linewidth=2)
    axes[0,1].plot(wavelengths_nm, R_broad, 'b-', label='Reflection', linewidth=2)
    axes[0,1].plot(wavelengths_nm, T_broad, 'g-', label='Transmission', linewidth=2)
    axes[0,1].set_title(f'Broadband Absorber (No Selectivity)\nReward: {reward_broad:.3f}')
    axes[0,1].set_xlabel('Wavelength (nm)')
    axes[0,1].set_ylabel('R/T/A')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Broadband reflector
    A_reflector = np.zeros_like(wavelengths)
    R_reflector = np.ones_like(wavelengths)
    T_reflector = np.zeros_like(wavelengths)
    reward_reflector = cal_reward_selective_1500(R_reflector, T_reflector, A_reflector, target, wavelengths)
    
    axes[0,2].plot(wavelengths_nm, A_reflector, 'r-', label='Absorption', linewidth=2)
    axes[0,2].plot(wavelengths_nm, R_reflector, 'b-', label='Reflection', linewidth=2)
    axes[0,2].plot(wavelengths_nm, T_reflector, 'g-', label='Transmission', linewidth=2)
    axes[0,2].set_title(f'Broadband Reflector (Penalized)\nReward: {reward_reflector:.3f}')
    axes[0,2].set_xlabel('Wavelength (nm)')
    axes[0,2].set_ylabel('R/T/A')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Partial transmission case
    A_partial = np.zeros_like(wavelengths)
    A_partial[absorption_window] = 0.7
    R_partial = np.ones_like(wavelengths) * 0.6
    R_partial[absorption_window] = 0.1
    T_partial = np.ones_like(wavelengths) * 0.4
    T_partial[absorption_window] = 0.2
    reward_partial = cal_reward_selective_1500(R_partial, T_partial, A_partial, target, wavelengths)
    
    axes[1,0].plot(wavelengths_nm, A_partial, 'r-', label='Absorption', linewidth=2)
    axes[1,0].plot(wavelengths_nm, R_partial, 'b-', label='Reflection', linewidth=2)
    axes[1,0].plot(wavelengths_nm, T_partial, 'g-', label='Transmission', linewidth=2)
    axes[1,0].set_title(f'Partial Transmission (Penalized)\nReward: {reward_partial:.3f}')
    axes[1,0].set_xlabel('Wavelength (nm)')
    axes[1,0].set_ylabel('R/T/A')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Narrow absorption peak
    narrow_window = (wavelengths >= 1.49) & (wavelengths <= 1.51)
    A_narrow = np.zeros_like(wavelengths)
    A_narrow[narrow_window] = 1.0
    R_narrow = np.ones_like(wavelengths)
    R_narrow[narrow_window] = 0.0
    T_narrow = np.zeros_like(wavelengths)
    reward_narrow = cal_reward_selective_1500(R_narrow, T_narrow, A_narrow, target, wavelengths)
    
    axes[1,1].plot(wavelengths_nm, A_narrow, 'r-', label='Absorption', linewidth=2)
    axes[1,1].plot(wavelengths_nm, R_narrow, 'b-', label='Reflection', linewidth=2)
    axes[1,1].plot(wavelengths_nm, T_narrow, 'g-', label='Transmission', linewidth=2)
    axes[1,1].set_title(f'Narrow Absorption Peak\nReward: {reward_narrow:.3f}')
    axes[1,1].set_xlabel('Wavelength (nm)')
    axes[1,1].set_ylabel('R/T/A')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvspan(1450, 1550, alpha=0.2, color='red')
    axes[1,1].axvspan(1490, 1510, alpha=0.3, color='orange', label='Actual absorption')
    
    # Shifted absorption peak
    shifted_window = (wavelengths >= 1.6) & (wavelengths <= 1.7)
    A_shifted = np.zeros_like(wavelengths)
    A_shifted[shifted_window] = 1.0
    R_shifted = np.ones_like(wavelengths)
    R_shifted[shifted_window] = 0.0
    T_shifted = np.zeros_like(wavelengths)
    reward_shifted = cal_reward_selective_1500(R_shifted, T_shifted, A_shifted, target, wavelengths)
    
    axes[1,2].plot(wavelengths_nm, A_shifted, 'r-', label='Absorption', linewidth=2)
    axes[1,2].plot(wavelengths_nm, R_shifted, 'b-', label='Reflection', linewidth=2)
    axes[1,2].plot(wavelengths_nm, T_shifted, 'g-', label='Transmission', linewidth=2)
    axes[1,2].set_title(f'Shifted Absorption Peak\nReward: {reward_shifted:.3f}')
    axes[1,2].set_xlabel('Wavelength (nm)')
    axes[1,2].set_ylabel('R/T/A')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].axvspan(1450, 1550, alpha=0.2, color='red')
    axes[1,2].axvspan(1600, 1700, alpha=0.3, color='orange', label='Actual absorption')
    
    # Weak absorption
    A_weak = np.zeros_like(wavelengths)
    A_weak[absorption_window] = 0.3
    R_weak = np.ones_like(wavelengths)
    R_weak[absorption_window] = 0.7
    T_weak = np.zeros_like(wavelengths)
    reward_weak = cal_reward_selective_1500(R_weak, T_weak, A_weak, target, wavelengths)
    
    axes[2,0].plot(wavelengths_nm, A_weak, 'r-', label='Absorption', linewidth=2)
    axes[2,0].plot(wavelengths_nm, R_weak, 'b-', label='Reflection', linewidth=2)
    axes[2,0].plot(wavelengths_nm, T_weak, 'g-', label='Transmission', linewidth=2)
    axes[2,0].set_title(f'Weak Absorption (30%)\nReward: {reward_weak:.3f}')
    axes[2,0].set_xlabel('Wavelength (nm)')
    axes[2,0].set_ylabel('R/T/A')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Multilayer interference
    A_interference, R_interference, T_interference, reward_interference = test_multilayer_interference()
    
    axes[2,1].plot(wavelengths_nm, A_interference, 'r-', label='Absorption', linewidth=2)
    axes[2,1].plot(wavelengths_nm, R_interference, 'b-', label='Reflection', linewidth=2)
    axes[2,1].plot(wavelengths_nm, T_interference, 'g-', label='Transmission', linewidth=2)
    axes[2,1].set_title(f'Multilayer Interference\nReward: {reward_interference:.3f}')
    axes[2,1].set_xlabel('Wavelength (nm)')
    axes[2,1].set_ylabel('R/T/A')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].axvspan(1450, 1550, alpha=0.2, color='red')
    
    # Realistic case
    np.random.seed(42)
    A_realistic = np.random.normal(0.05, 0.02, len(wavelengths))
    A_realistic[absorption_window] = np.random.normal(0.85, 0.05, np.sum(absorption_window))
    A_realistic = np.clip(A_realistic, 0, 1)
    
    R_realistic = np.random.normal(0.85, 0.05, len(wavelengths))
    R_realistic[absorption_window] = np.random.normal(0.1, 0.05, np.sum(absorption_window))
    R_realistic = np.clip(R_realistic, 0, 1)
    
    T_realistic = np.random.normal(0.05, 0.02, len(wavelengths))
    T_realistic = np.clip(T_realistic, 0, 1)
    
    total = A_realistic + R_realistic + T_realistic
    A_realistic /= total
    R_realistic /= total
    T_realistic /= total
    
    reward_realistic = cal_reward_selective_1500(R_realistic, T_realistic, A_realistic, target, wavelengths)
    
    axes[2,2].plot(wavelengths_nm, A_realistic, 'r-', label='Absorption', linewidth=2)
    axes[2,2].plot(wavelengths_nm, R_realistic, 'b-', label='Reflection', linewidth=2)
    axes[2,2].plot(wavelengths_nm, T_realistic, 'g-', label='Transmission', linewidth=2)
    axes[2,2].set_title(f'Realistic Multilayer Response\nReward: {reward_realistic:.3f}')
    axes[2,2].set_xlabel('Wavelength (nm)')
    axes[2,2].set_ylabel('R/T/A')
    axes[2,2].legend()
    axes[2,2].grid(True, alpha=0.3)
    axes[2,2].axvspan(1450, 1550, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig('reward_test_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'reward_test_visualization.png'")


def run_all_tests():
    """Run all test cases and summarize results"""
    print("="*60)
    print("REWARD FUNCTION TESTING FOR SELECTIVE 1500nm ABSORBER")
    print("="*60)
    
    results = {}
    
    try:
        results['perfect'] = test_perfect_case()
        print()
        
        results['worst'] = test_worst_case()
        print()
        
        results['broadband_absorber'] = test_broadband_absorber()
        print()
        
        results['broadband_reflector'] = test_broadband_reflector()
        print()
        
        results['partial_transmission'] = test_partial_transmission()
        print()
        
        results['narrow_peak'] = test_narrow_absorption_peak()
        print()
        
        results['shifted_peak'] = test_shifted_absorption_peak()
        print()
        
        results['weak_absorption'] = test_weak_absorption()
        print()
        
        results['sharp_edge'], results['gradual_edge'] = test_edge_sharpness()
        print()
        
        results['no_wavelength'] = test_no_wavelength_info()
        print()
        
        results['realistic'] = test_realistic_case()
        print()
        
        # Get multilayer interference result
        _, _, _, results['multilayer_interference'] = test_multilayer_interference()
        print()
        
        visualize_test_cases()
        print()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    
    # Summarize results
    print("="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, reward in results.items():
        if isinstance(reward, tuple):
            print(f"{test_name:20s}: {reward[0]:.4f}, {reward[1]:.4f}")
        else:
            print(f"{test_name:20s}: {reward:.4f}")
    
    print()
    print("EXPECTED BEHAVIOR:")
    print("- Perfect case should have highest reward (>4.0)")
    print("- Worst case should have negative reward")
    print("- Broadband absorber should have lower reward than perfect (no selectivity)")
    print("- Broadband reflector should have very low/negative reward (no absorption)")
    print("- Partial transmission should be heavily penalized")
    print("- Shifted/narrow peaks should be penalized for poor target matching")
    print("- Weak absorption should be penalized for insufficient absorption")
    print("- Sharp edges should be rewarded over gradual edges")
    print("- Realistic cases should have moderate positive rewards")
    
    # Validation checks
    validation_passed = True
    
    if results['perfect'] <= 4.0:
        print("❌ FAIL: Perfect case reward too low")
        validation_passed = False
    else:
        print("✅ PASS: Perfect case has high reward")
    
    if results['worst'] >= 0:
        print("❌ FAIL: Worst case reward should be negative")
        validation_passed = False
    else:
        print("✅ PASS: Worst case has negative reward")
    
    if results['broadband_absorber'] >= results['perfect']:
        print("❌ FAIL: Broadband absorber should have lower reward than perfect")
        validation_passed = False
    else:
        print("✅ PASS: Broadband absorber properly penalized for lack of selectivity")
    
    if results['broadband_reflector'] >= 0:
        print("❌ FAIL: Broadband reflector should have negative/very low reward")
        validation_passed = False
    else:
        print("✅ PASS: Broadband reflector properly penalized for no absorption")
    
    if results['partial_transmission'] >= results['perfect']:
        print("❌ FAIL: Partial transmission should be penalized")
        validation_passed = False
    else:
        print("✅ PASS: Partial transmission properly penalized")
    
    if results['shifted_peak'] >= results['perfect'] * 0.8:
        print("❌ FAIL: Shifted peak should be significantly penalized")
        validation_passed = False
    else:
        print("✅ PASS: Shifted peak properly penalized")
    
    if results['weak_absorption'] >= results['perfect'] * 0.8:
        print("❌ FAIL: Weak absorption should be penalized")
        validation_passed = False
    else:
        print("✅ PASS: Weak absorption properly penalized")
    
    print()
    if validation_passed:
        print("🎉 ALL TESTS PASSED! Reward function behaves as expected.")
    else:
        print("⚠️  SOME TESTS FAILED! Review reward function implementation.")
    
    return validation_passed


if __name__ == "__main__":
    run_all_tests()
"""Quick test for broadband reflector penalty fix"""

import numpy as np
from RLMultilayer.utils import cal_reward_selective_1500

def test_broadband_reflector_fix():
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
    print(f"Broadband reflector reward (should be negative): {reward:.4f}")
    
    if reward < 0:
        print("✅ FIXED: Broadband reflector now properly penalized!")
        return True
    else:
        print("❌ Still needs fixing")
        return False

if __name__ == "__main__":
    test_broadband_reflector_fix()
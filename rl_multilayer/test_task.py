"""
Simple test script to verify the new 1500nm selective absorber task works correctly.
"""

import numpy as np
from RLMultilayer.taskenvs.tasks import get_env_fn

def test_task_creation():
    """Test that the task can be created successfully"""
    print("Testing erent_1500_absorber_task_v0 creation...")
    
    try:
        # Create environment function
        env_fn = get_env_fn('erent_1500_absorber_task_v0', discrete_thick=True, bottom_up=False, spectrum_repr=False)
        
        # Create environment instance
        env = env_fn()
        
        print("✅ Task created successfully!")
        print(f"   Materials: {env.materials}")
        print(f"   Wavelength range: {env.simulator.wavelength[0]:.2f} - {env.simulator.wavelength[-1]:.2f} μm")
        print(f"   Number of wavelength points: {len(env.simulator.wavelength)}")
        print(f"   Thickness options: {len(env.thickness_list)} steps from {env.thickness_list[0]} to {env.thickness_list[-1]} nm")
        print(f"   Target absorption window: 1450-1550 nm")
        
        # Test a simple environment step
        print("\nTesting environment interaction...")
        
        # Reset environment
        obs = env.reset()
        print(f"   Initial observation shape: {obs.shape}")
        
        # Test a few actions
        # Action format: [material_index, thickness_index]
        test_actions = [
            [0, 10],  # First material, moderate thickness
            [1, 20],  # Second material, different thickness
            [len(env.materials), 0]  # EOS (End of Structure) action
        ]
        
        total_reward = 0
        for i, action in enumerate(test_actions):
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"   Step {i+1}: Action {action}, Reward: {reward:.4f}, Done: {done}")
            
            if done:
                print(f"   Episode finished with total reward: {total_reward:.4f}")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating task: {e}")
        return False

def test_reward_function():
    """Test the reward function with a known structure"""
    print("\nTesting reward function behavior...")
    
    try:
        from RLMultilayer.utils import cal_reward_selective_1500
        
        # Create test wavelengths
        wavelengths = np.arange(0.4, 2.1, 0.01)
        
        # Test perfect selective absorber
        absorption_window = (wavelengths >= 1.45) & (wavelengths <= 1.55)
        
        A_perfect = np.zeros_like(wavelengths)
        A_perfect[absorption_window] = 1.0
        
        R_perfect = np.ones_like(wavelengths)
        R_perfect[absorption_window] = 0.0
        
        T_perfect = np.zeros_like(wavelengths)
        
        target = {'A': A_perfect, 'R': R_perfect, 'T': T_perfect}
        
        reward = cal_reward_selective_1500(R_perfect, T_perfect, A_perfect, target, wavelengths)
        
        print(f"   Perfect case reward: {reward:.4f}")
        
        if reward > 4.0:
            print("✅ Reward function working correctly!")
            return True
        else:
            print("❌ Reward function may have issues")
            return False
            
    except Exception as e:
        print(f"❌ Error testing reward function: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING ERENT 1500nm ABSORBER TASK")
    print("="*60)
    
    task_test = test_task_creation()
    reward_test = test_reward_function()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if task_test and reward_test:
        print("🎉 ALL TESTS PASSED! The task is ready for RL training.")
        print("\nTo use this task in your RL training:")
        print("   env_name = 'erent_1500_absorber_task_v0'")
        print("   env_fn = get_env_fn(env_name, discrete_thick=True, bottom_up=False)")
        print("   env = env_fn()")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")

if __name__ == "__main__":
    main()
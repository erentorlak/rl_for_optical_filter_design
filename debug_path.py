import os
import sys

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

# Test different approaches
approaches = [
    ("Direct relative", os.path.join("data", "Cr.csv")),
    ("Dot relative", os.path.join(".", "data", "Cr.csv")),
    ("Parent then data", os.path.join("..", "data", "Cr.csv")),
]

for name, path in approaches:
    exists = os.path.exists(path)
    abs_path = os.path.abspath(path)
    print(f"{name:15} | {path:20} | {exists} | {abs_path}")

# Test the exact path that utils.py would calculate
print("\nSimulating utils.py calculation:")
utils_path = os.path.abspath(os.path.join("rl_multilayer", "RLMultilayer", "utils.py"))
print(f"Utils file at: {utils_path}")

_current_file = utils_path
_rl_multilayer_dir = os.path.dirname(os.path.dirname(_current_file))
_project_root = os.path.dirname(_rl_multilayer_dir)
_absolute_data_path = os.path.join(_project_root, 'data')

print(f"Current file: {_current_file}")
print(f"RL multilayer dir: {_rl_multilayer_dir}")
print(f"Project root: {_project_root}")
print(f"Data path: {_absolute_data_path}")
print(f"Data exists: {os.path.exists(_absolute_data_path)}")
print(f"Cr.csv exists: {os.path.exists(os.path.join(_absolute_data_path, 'Cr.csv'))}")
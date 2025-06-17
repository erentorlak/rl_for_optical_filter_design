import os

# This simulates the path calculation from utils.py
# Current file: /rl-optical-design-dev1/rl_multilayer/RLMultilayer/utils.py
# We need: /rl-optical-design-dev1/data

# Simulate the path from utils.py location
utils_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_multilayer', 'RLMultilayer', 'utils.py')
print(f"Utils file would be at: {utils_file_path}")

_current_dir = os.path.dirname(utils_file_path)  # /rl_multilayer/RLMultilayer/
_rl_multilayer_dir = os.path.dirname(_current_dir)         # /rl_multilayer/
_project_root = os.path.dirname(_rl_multilayer_dir)        # /rl-optical-design-dev1/
DATABASE = os.path.join(_project_root, 'data')

print(f"Current dir: {_current_dir}")
print(f"RL multilayer dir: {_rl_multilayer_dir}")
print(f"Project root: {_project_root}")
print(f"Database path: {DATABASE}")
print(f"Cr.csv path: {os.path.join(DATABASE, 'Cr.csv')}")
print(f"Cr.csv exists: {os.path.exists(os.path.join(DATABASE, 'Cr.csv'))}")

# Also test simple relative path
simple_path = os.path.join('.', 'data', 'Cr.csv')
print(f"Simple relative path: {simple_path}")
print(f"Simple path exists: {os.path.exists(simple_path)}")
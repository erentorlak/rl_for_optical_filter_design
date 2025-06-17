import pandas as pd
import os

data_dir = 'C:/zzaaa/rl-optical-design-dev1/data'   

# Check wavelength ranges for all materials
def get_material_names_from_data_folder(data_dir):
    material_names = set()
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                material_name = os.path.splitext(file)[0]
                material_names.add(material_name)
    return sorted(list(material_names))


for material in get_material_names_from_data_folder(data_dir):
    file_path = os.path.join(data_dir, material + '.csv')
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"{material}: {data['wl'].min()} - {data['wl'].max()} μm")
    else:
        print(f"{material}: File not found")
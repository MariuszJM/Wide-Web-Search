import os
import yaml
from datetime import datetime


def create_output_directory(base_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results(processed_items, output_dir):
    top_items = processed_items.get('top_items', {})
    with open(os.path.join(output_dir, 'top_items.yaml'), 'w') as f:
        yaml.dump(top_items, f, default_flow_style=False, sort_keys=False)
    less_relevant_items = processed_items.get('less_relevant_items', {})
    with open(os.path.join(output_dir, 'less_relevant_items.yaml'), 'w') as f:
        yaml.dump(less_relevant_items, f, default_flow_style=False, sort_keys=False)

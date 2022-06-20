import numpy as np
import yaml
from dataset import CustomDataset
from ysdc_dataset_api.features import FeatureRenderer
from trajectory_set_generator import FixedGenerator

def filter_stationary_trajectory(trajectory_tags_list):
    return 'kStationary' not in trajectory_tags_list

if __name__ == "__main__":
    
    with open("yandex_shifts_covernet.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        renderer_config = config['renderer']
    renderer = FeatureRenderer(renderer_config)
    
    root = '/mnt/Vol0/datasets/yandex_shifts/sdc/data'

    dataset_path = root + '/train_pb'
    prerendered_dataset_path = root + '/train_rendered'
    scene_tags_fpath = root + '/train_tags.txt'

    past_steps = 1
    
    dataset = CustomDataset(
        dataset_path=dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        feature_producers=[renderer],
        transform_ground_truth_to_agent_frame=True,
        prerendered_dataset_path=prerendered_dataset_path,
        past_steps = past_steps,
        trajectory_tags_filter=filter_stationary_trajectory,
        div='moving'
    )
    
    fg = FixedGenerator(num_modes=64, dataset=dataset, epsilon=2, limit=15000, renderer=renderer)
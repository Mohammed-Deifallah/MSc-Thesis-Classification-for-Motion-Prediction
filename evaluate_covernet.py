import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from ysdc_dataset_api.features import FeatureRenderer
from trajectory_set_generator import FixedGenerator
from metrics import ADE, FDE
import pickle
from tqdm import tqdm
import yaml

if __name__ == "__main__":
    
    def _init_fn(worker_id):
        np.random.seed(42)
    
    def evaluate(criterion):
        model = torch.load(model_path, map_location=device)
        model.eval()
        cum_loss = 0

        with torch.no_grad():
            for item in tqdm(val_dataloader):
                hdmap = item['feature_maps'].to(device)
                agent_state_vector = item['agent_state_vector'].to(device)
                ground_truth = item['ground_truth_trajectory'].to(device)

                output = model(hdmap, agent_state_vector)
                loss = criterion(output, ground_truth)
                cum_loss += loss.item()

        return cum_loss / len(val_dataloader)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=int, default=2, help='error tolerance between trajectories to consider close')
    parser.add_argument('-k', type=int, default=64, help='number of modes')
    parser.add_argument('--saved_model', type=str, help='path to weights of a pretrained model')

    args = parser.parse_args()
    
    with open("yandex_shifts_covernet.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        
    renderer = FeatureRenderer(config['renderer'])
    
    root = '/mnt/Vol0/datasets/yandex_shifts/sdc/data'

    val_dataset_path = root + '/development_pb'
    prerendered_val_dataset_path = root + '/development_rendered'
    val_scene_tags_fpath = root + '/development_tags.txt'

    save_root_path = '/hdd/CoverNet/saved_models'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    val_dataset = CustomDataset(
        dataset_path=val_dataset_path,
        scene_tags_fpath=val_scene_tags_fpath,
        feature_producers=[renderer],
        transform_ground_truth_to_agent_frame=True,
        prerendered_dataset_path=prerendered_val_dataset_path,
        div='val',
        limit=50000
    )
    
    eps, k = args.e, args.k
    fg = FixedGenerator(load_fpth=f'epsilon_{eps}_k_{k}.pkl')
    traj_set = fg.traj_set
    
    model_path = f'{save_root_path}/{args.saved_model}'

    val_dataloader = DataLoader(val_dataset, batch_size=256, worker_init_fn=_init_fn, pin_memory=True)
    
    ade_loss = evaluate(ADE(traj_set))
    fde_loss = evaluate(FDE(traj_set))
    
    with open(f'{save_root_path}/EVAL_CLOSEST_{args.saved_model[:-3]}.pkl', "wb") as handle:
        pickle.dump({'ADE': ade_loss, 'FDE': fde_loss}, handle, protocol=pickle.HIGHEST_PROTOCOL)
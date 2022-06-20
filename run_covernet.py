import argparse

import os
import numpy as np
import yaml
import torch
import ysdc_dataset_api
from ysdc_dataset_api.utils import get_file_paths, scenes_generator, transform_2d_points
from ysdc_dataset_api.dataset import MotionPredictionDataset
from dataset import CustomDataset
from ysdc_dataset_api.features import FeatureRenderer
from trajectory_set_generator import FixedGenerator
from covernet_backbone import ResNetBackbone, MobileNetBackbone
from covernet_predictor import CoverNet
from metrics import ConstantLatticeLoss
from train import train
import pickle

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=int, default=1, help='number of the past steps of the agent history')
    parser.add_argument('-e', type=int, default=2, help='error tolerance between trajectories to consider close')
    parser.add_argument('-k', type=int, default=64, help='number of modes')
    parser.add_argument('-b', type=str, default='resnet50', help='backbone type: [\'resnet50\', \'mobilenet_v2\'')
    parser.add_argument('--pretrained', action='store_true', help='use the pretrained backbone from PyTorch')
    parser.add_argument('--saved_backbone', type=str, help='path to weights of a pretrained backbone')
    parser.add_argument('--saved_model', type=str, help='path to weights of a pretrained model')
    

    args = parser.parse_args()
    

    with open("yandex_shifts_covernet.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        
    renderer = FeatureRenderer(config['renderer'])
    
    root = '/mnt/Vol0/datasets/yandex_shifts/sdc/data'

    dataset_path = root + '/train_pb'
    prerendered_dataset_path = root + '/train_rendered'
    scene_tags_fpath = root + '/train_tags.txt'

    val_dataset_path = root + '/development_pb'
    prerendered_val_dataset_path = root + '/development_rendered'
    val_scene_tags_fpath = root + '/development_tags.txt'

    save_root_path = '/hdd/CoverNet/saved_models'

    past_steps = args.p

    dataset = CustomDataset(
        dataset_path=dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        feature_producers=[renderer],
        transform_ground_truth_to_agent_frame=True,
        prerendered_dataset_path=prerendered_dataset_path,
        past_steps = past_steps,
        #trajectory_tags_filter=filter_stationary_trajectory,
        limit=500000,
        div='train'
    )
    
    val_dataset = CustomDataset(
        dataset_path=val_dataset_path,
        scene_tags_fpath=val_scene_tags_fpath,
        feature_producers=[renderer],
        transform_ground_truth_to_agent_frame=True,
        prerendered_dataset_path=prerendered_val_dataset_path,
        past_steps = past_steps,
        limit=200000,
        div='val'
    )
    
    eps, k = args.e, args.k
    fg = FixedGenerator(load_fpth=f'epsilon_{eps}_k_{k}.pkl')
    traj_set = fg.traj_set
    
    backbone_path = None
    if args.saved_backbone is not None:
        backbone_path = f'{save_root_path}/{args.saved_backbone}'
    
    model_path = None
    if args.saved_model is not None:
        model_path = f'{save_root_path}/{args.saved_model}'
    
    pretrained = args.pretrained and backbone_path is None and model_path is None

    if args.b == 'mobilenet_v2':
        backbone = MobileNetBackbone((17, 128, 128), pretrained=pretrained, weights_path=backbone_path)
    else:
        backbone = ResNetBackbone((17, 128, 128), pretrained=pretrained, weights_path=backbone_path)
        if args.b != 'resnet50':
            args.b = 'resnet50'
            print(f'A wrong value is passed ({args.b}). ResNet50 will be used!')
    
    if model_path is None:
        covernet_model = CoverNet(backbone=backbone, asv_dim=5*past_steps, num_modes=k)
    else:
        covernet_model = torch.load(model_path, map_location=device)
    
    naming_suffix = f"fixed_eps_{eps}_k_{k}{'_pretrained' if pretrained else ''}_{args.b}"\
    f"{'_' + args.saved_backbone[:-3] if args.saved_backbone is not None else ''}"\
    f"{'_' + args.saved_model[:-3] if args.saved_model is not None else ''}"
    
    loss, val_loss = train(dataset, val_dataset, covernet_model, ConstantLatticeLoss(traj_set), 
          f"{save_root_path}/{naming_suffix}")
    
    with open(f'{save_root_path}/LOSS_{naming_suffix}.pkl', "wb") as handle:
        pickle.dump([loss, val_loss], handle, protocol=pickle.HIGHEST_PROTOCOL)
from typing import List
from tqdm import tqdm
import numpy as np
import pickle
import os
import torch
from ysdc_dataset_api.utils import transform_2d_points
from ysdc_dataset_api.features import FeatureRenderer

class Graph:
    
    def __init__(self, n: int):
        self.neighbors = [[] for i in range(n)]
        self.degrees = [(i, 0) for i in range(n)]
        
    def add_edge(self, idx1: int, idx2: int) -> None:        
        self.neighbors[idx1].append(idx2)
        self.neighbors[idx2].append(idx1)
        
        self.degrees[idx1] = (self.degrees[idx1][0], self.degrees[idx1][1] + 1)
        self.degrees[idx2] = (self.degrees[idx2][0], self.degrees[idx2][1] + 1)
    

class Generator:
    """This class is parent class for all types of trajectory set generators
    """
    def __init__(self, num_modes: int=64):
        self.num_modes = num_modes
        self.traj_set = []
    
    def _resolve(self):
        pass
    
    @staticmethod
    def max_pointwise_l2_norm(tj1: np.ndarray, tj2: np.ndarray) -> float:
        norm = 0
        for i in range(len(tj1)):
            norm = max(norm, np.linalg.norm(tj1[i] - tj2[i]))
        return norm
    

class FixedGenerator(Generator):
    
    def __init__(self, num_modes: int=64, dataset: torch.utils.data.IterableDataset=None, load_traj: str=None,
                 epsilon: int=2, save_fpth: str=None, load_fpth: str=None, limit: int=None, renderer: FeatureRenderer=None):
        super().__init__(num_modes)
        
        if load_fpth is not None:
            self.traj_set = np.load(load_fpth, allow_pickle=True)
        
        else:
            self.eps = epsilon
            if save_fpth is None:
                save_fpth = f'./epsilon_{self.eps}_k_{self.num_modes}.pkl'
                
            if load_traj is None:
                load_traj = 'traj.pkl'
                
            if os.path.isfile(load_traj):
                self.trajectories = np.load(load_traj, allow_pickle=True)
                
            else:
                dataset.build_fixed_set = True
                self.trajectories = []
                for item in tqdm(iter(dataset)):
                    traj = item['ground_truth_trajectory']
                    if renderer is not None:
                        traj = transform_2d_points(traj, renderer.to_feature_map_tf)
                        traj = np.round(traj - 0.5).astype(np.int32)
                    self.trajectories.append(traj)
                with open('traj.pkl', 'wb') as handle:
                    pickle.dump(self.trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                dataset.build_fixed_set = False
            
            if limit is not None:
                np.random.seed(42)
                indices = np.random.choice(len(self.trajectories), limit, replace=False)
                self.trajectories = [self.trajectories[i] for i in indices]
            
            self.traj_set = [np.array([[64, 64] for _ in range(len(self.trajectories[0]))])]
            if len(self.trajectories) <= self.num_modes:
                self.traj_set = self.trajectories
            else:
                self._resolve()
            with open(save_fpth, 'wb') as handle:
                pickle.dump(self.traj_set, handle, protocol=pickle.HIGHEST_PROTOCOL)            
    
    def _resolve(self):
        n = len(self.trajectories)
        g = Graph(n)
        
        if os.path.isfile(f'degrees_{self.eps}_{n}.pkl'):
            g.degrees = np.load(f'degrees_{self.eps}_{n}.pkl', allow_pickle=True)
            g.neighbors = np.load(f'neighbors_{self.eps}_{n}.pkl', allow_pickle=True)
        
        else:
            for i in tqdm(range(n)):
                for j in range(i + 1, n):
                    if Generator.max_pointwise_l2_norm(self.trajectories[i], self.trajectories[j]) <= self.eps:
                        g.add_edge(i, j)

            with open(f'degrees_{self.eps}_{n}.pkl', 'wb') as handle:
                pickle.dump(g.degrees, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'neighbors_{self.eps}_{n}.pkl', 'wb') as handle:
                pickle.dump(g.neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        vertices = set()
        degrees = sorted(g.degrees, key=lambda x: x[1])
        
        while len(self.traj_set) < self.num_modes:
            
            idx = degrees.pop()[0]
            if idx in vertices:
                continue
            
            vertices.add(idx)
            self.traj_set.append(self.trajectories[idx])
            for k in range(len(g.neighbors[idx])):
                neigh_idx = g.neighbors[idx][k]
                
                if neigh_idx in vertices:
                    continue
                vertices.add(neigh_idx)
                
        return

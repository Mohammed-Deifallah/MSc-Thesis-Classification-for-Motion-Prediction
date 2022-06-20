import numpy as np
import torch
import ysdc_dataset_api
from typing import Callable, Union, Optional, List
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureProducerBase
from ysdc_dataset_api.proto import Scene, get_tags_from_request, proto_to_dict
from ysdc_dataset_api.utils import (
    get_gt_trajectory,
    get_latest_track_state_by_id,
    get_to_track_frame_transform,
    request_is_valid,
    scenes_generator,
    transform_2d_points,
    read_feature_map_from_file
)
from tqdm import tqdm
import os
from utils import *


class CustomDataset(MotionPredictionDataset):
    def __init__(
            self,
            dataset_path: str,
            scene_tags_fpath: str = None,
            feature_producers: List[FeatureProducerBase] = None,
            prerendered_dataset_path: str = None,
            transform_ground_truth_to_agent_frame: bool = True,
            scene_tags_filter: Union[Callable, None] = None,
            trajectory_tags_filter: Union[Callable, None] = None,
            pre_filtered_scene_file_paths: Optional[List[str]] = None,
            yield_metadata=True,
            past_steps: int = 1,
            div: str = 'train',
            limit: int = -1
    ):
        """Pytorch-style dataset class for the motion prediction task.

        Dataset iterator performs iteration over scenes in the dataset and individual prediction
        requests in each scene. Iterator yields dict that can have the following structure:
        {
            'scene_id': str,
            'track_id': int,
            'scene_tags': Dict[str, str],
            'ground_truth_trajectory': np.ndarray,
            'agent_state_vector': np.ndarray,
            'prerendered_feature_map': np.ndarray,
            'feature_maps': np.ndarray,
        }.
        'scene_id' unique scene identifier.
        'track_id' vehicle id of the current prediction request.
        'ground_truth_trajectory' the field contains ground truth trajectory for
        the current prediction request.
        'agent_state_vector' the field contains velocity, acceleration and yaw rate for agents.
        'prerendered_feature_map' field would be present if prerendered_dataset_path was specified,
        contains pre-rendered feature maps.
        'feature_maps' field would be present if user passes an instance of
        ysdc_dataset_api.features.FeatureRenderer, contains feature maps rendered on the fly by
        specified renderer instance.

        Args:
            dataset_path: path to the dataset directory
            scene_tags_fpath: path to the tags file
            feature_producer: instance of the FeatureProducerBase class,
                used to generate features for a data item. Defaults to None.
            prerendered_dataset_path: path to the pre-rendered dataset. Defaults to None.
            transform_ground_truth_to_agent_frame: whether to transform ground truth
                trajectory to an agent coordinate system or return global coordinates.
                Defaults to True.
            scene_tags_filter: function to filter dataset scenes by tags. Defaults to None.
            trajectory_tags_filter: function to filter prediction requests by trajectory tags.
                Defaults to None.
            past_steps: the number of past steps extracted from the agent history, Defaults to 1.
            div: the type of the dataset. This helps return the appropriate length.
            limit: the number of data items to extract from the datases. -1 mean the full dataset.

        Raises:
            ValueError: if none of feature_producer or prerendered_dataset_path was specified.
        """
        super().__init__(dataset_path, scene_tags_fpath, feature_producers,
            prerendered_dataset_path, transform_ground_truth_to_agent_frame, scene_tags_filter,
            trajectory_tags_filter, pre_filtered_scene_file_paths, yield_metadata)
        self.past_steps = past_steps
        self.build_fixed_set = False
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.file_paths = self._scene_file_paths
        else:
            self.file_paths = self._split_filepaths_by_worker(
                worker_info.id, worker_info.num_workers)
            
        if div == 'train':
            self.length = 5649675
            if limit > 0:
                self.length = min(limit, 5649675)
        elif div == 'val':
            self.length = 465421
            if limit > 0:
                self.length = min(limit, 465421)
        elif div == 'moving':
            self.length = 3324459
            if limit > 0:
                self.length = min(limit, 3324459)
        else:
            self.length = 0

            for scene, fpath in tqdm(scenes_generator(self.file_paths, yield_fpath=True)):
                for request in scene.prediction_requests:
                    if not request_is_valid(scene, request):
                        continue
                    trajectory_tags = get_tags_from_request(request)
                    if not self._trajectory_tags_filter(trajectory_tags):
                        continue
                    self.length += 1
                    if self.length == limit:
                        break
                if self.length == limit:
                    break
    
    def __len__(self):
        return self.length

    def __iter__(self):
                
        return data_generator(
            self.file_paths,
            self._feature_producers,
            self._prerendered_dataset_path,
            self._transform_ground_truth_to_agent_frame,
            self._trajectory_tags_filter,
            self._yield_metadata,
            yield_scene_tags=True,
            past_steps=self.past_steps,
            build_fixed_set=self.build_fixed_set,
            limit=self.length
        )
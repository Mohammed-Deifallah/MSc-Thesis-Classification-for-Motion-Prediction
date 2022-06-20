import os
from typing import Callable, List

import numpy as np

from ysdc_dataset_api.features import FeatureProducerBase, FeatureRenderer
from sdc.constants import SCENE_TAG_TYPE_TO_OPTIONS, VALID_TRAJECTORY_TAGS
from ysdc_dataset_api.proto import Scene, get_tags_from_request, proto_to_dict
from ysdc_dataset_api.utils import (
    get_gt_trajectory,
    get_latest_track_state_by_id,
    get_to_track_frame_transform,
    read_feature_map_from_file,
    request_is_valid,
    scenes_generator,
    transform_2d_points,
)



def _add_metadata_to_batch(scene, request, trajectory_tags, batch):
    batch['scene_id'] = scene.id
    batch['request_id'] = request.track_id

    # Note that some will be "invalid"
    batch['num_vehicles'] = len(scene.prediction_requests)

    scene_tags_dict = proto_to_dict(scene.scene_tags)
    for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
        scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]

        for scene_tag_option in scene_tag_options:
            try:
                batch[f'{scene_tag_type}__{scene_tag_option}'] = int(
                    scene_tags_dict[scene_tag_type] == scene_tag_option)
            except KeyError:
                batch[f'{scene_tag_type}__{scene_tag_option}'] = -1

    trajectory_tags = set(trajectory_tags)
    for trajectory_tag in VALID_TRAJECTORY_TAGS:
        batch[trajectory_tag] = (trajectory_tag in trajectory_tags)

    return batch

def _get_agent_state_vector(scene: Scene, track_id: int, past_steps: int) -> np.ndarray:
        """Extracts a state vector from scene for an object with track_id.

        Args:
            scene (Scene): scene to extract state vector from
            track_id (int): track id to extract state vector for
            past_steps (int): the number of past steps extracted from the agent history, Defaults to 1.
        Returns:
            asv (np.ndarray): array of shape (past_steps, 5)
        """
        asv = np.empty((0, 5), dtype=np.float32)
        i = 0
        for i in range(past_steps):
            for track in scene.past_vehicle_tracks[-(i + 1)].tracks:
                if track.track_id == track_id:
                    asv = np.vstack([asv, [track.linear_velocity.x, track.linear_velocity.y, 
                              track.linear_acceleration.x, track.linear_acceleration.y, 
                              track.yaw]]).astype(np.float32)
                    break
            
            if asv.shape[0] == i:
                break
        return asv

def _get_serialized_fm_path(prerendered_dataset_path, scene_fpath, scene_id, track_id):
    base, _ = os.path.split(scene_fpath)
    _, subdir = os.path.split(base)
    return os.path.join(prerendered_dataset_path, subdir, f'{scene_id}_{track_id}.npy')
    
def data_generator(
    file_paths: List[str],
    feature_producers: List[FeatureProducerBase],
    prerendered_dataset_path: str,
    transform_ground_truth_to_agent_frame: bool,
    trajectory_tags_filter: Callable,
    yield_metadata: bool,
    yield_scene_tags: bool,
    past_steps: int,
    build_fixed_set: bool,
    limit: int
    ):
    
    renderer = None
    for rend in feature_producers:
        if isinstance(rend, FeatureRenderer):
            renderer = rend
            break
    
    cnt = 0
    for scene, fpath in scenes_generator(file_paths, yield_fpath=True):
        for request in scene.prediction_requests:
            if cnt == limit:
                return
            
            if not request_is_valid(scene, request):
                continue
            trajectory_tags = get_tags_from_request(request)
            if not trajectory_tags_filter(trajectory_tags):
                continue
            track = get_latest_track_state_by_id(scene, request.track_id)
            to_track_frame_tf = get_to_track_frame_transform(track)

            result = {
                'scene_id': scene.id,
                'track_id': request.track_id,
            }
            
            if yield_scene_tags:
                result['scene_tags'] = proto_to_dict(scene.scene_tags)

            ground_truth_trajectory = get_gt_trajectory(scene, request.track_id)
            if ground_truth_trajectory.shape[0] > 0:
                if transform_ground_truth_to_agent_frame:
                    ground_truth_trajectory = transform_2d_points(
                        ground_truth_trajectory, to_track_frame_tf)
            
                if renderer is not None:
                        ground_truth_trajectory = transform_2d_points(ground_truth_trajectory, renderer.to_feature_map_tf)
                        ground_truth_trajectory = np.round(ground_truth_trajectory - 0.5).astype(np.int32)
                        
                result['ground_truth_trajectory'] = ground_truth_trajectory

            agent_state_vector = _get_agent_state_vector(scene, request.track_id, past_steps)
            if agent_state_vector.shape[0] > 0 and transform_ground_truth_to_agent_frame:
                agent_state_vector[:, :2] = transform_2d_points(agent_state_vector[:, :2], to_track_frame_tf)
                agent_state_vector[:, 2:4] = transform_2d_points(agent_state_vector[:, 2:4], to_track_frame_tf)

            agent_state_vector = agent_state_vector.flatten()
            result['agent_state_vector'] = np.pad(agent_state_vector,
                                                  pad_width=(0, 5 * past_steps - len(agent_state_vector)),
                                                  mode='constant')

            if not build_fixed_set:
                if prerendered_dataset_path:
                    fm_path = _get_serialized_fm_path(
                        prerendered_dataset_path, fpath, scene.id, request.track_id)
                    result['prerendered_feature_map'] = read_feature_map_from_file(fm_path)

                for producer in feature_producers:
                    result.update(producer.produce_features(scene, request))

                if yield_metadata:
                    result = _add_metadata_to_batch(
                        scene=scene, request=request,
                        trajectory_tags=trajectory_tags,
                        batch=result)
            
            cnt += 1
            yield result
#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def drop_feature_names(features: dict, names: list[str]) -> dict[str, list[int]]:
    """Removes `names` (e.g. "head_motor_1.pos") from every named vector feature such as observation.state and
    action, updating its names and shape in place. Returns the kept indices of each feature that changed."""
    keep = {}
    for key, ft in features.items():
        ft_names = ft.get("names")
        if isinstance(ft_names, list) and set(ft_names) & set(names):
            keep[key] = [i for i, n in enumerate(ft_names) if n not in names]
            ft["names"] = [ft_names[i] for i in keep[key]]
            ft["shape"] = (len(keep[key]),)
    return keep


class DropNamesDataset(torch.utils.data.Dataset):
    """A LeRobotDataset without some named state/action entries (see drop_feature_names): items, meta.features
    and the per-dimension meta.stats are all sliced, so the policy and its saved normalizer never see them."""

    def __init__(self, dataset: LeRobotDataset, names: list[str]):
        self.dataset = dataset
        sizes = {k: len(ft["names"]) for k, ft in dataset.meta.features.items() if isinstance(ft.get("names"), list)}
        self.keep = drop_feature_names(dataset.meta.features, names)
        for key, idx in self.keep.items():
            stats = dataset.meta.stats.get(key, {})
            for stat, value in stats.items():
                if np.shape(value)[-1:] == (sizes[key],):  # per-dimension stats ("count" is a single number)
                    stats[stat] = value[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]
        for key, keep in self.keep.items():
            item[key] = item[key][..., keep]
        return item

    def __getattr__(self, name):  # meta, num_frames, episodes, ... come from the wrapped dataset
        if name == "dataset":  # not set yet while a DataLoader worker unpickles this object
            raise AttributeError(name)
        return getattr(self.dataset, name)


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    tf_cfg = cfg.dataset.image_transforms
    # With on_gpu the training loop applies them instead.
    image_transforms = ImageTransforms(tf_cfg) if tf_cfg.enable and not tf_cfg.on_gpu else None

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.drop_names:
        if cfg.dataset.streaming:
            raise ValueError("--dataset.drop_names needs a non-streaming dataset; pass --dataset.drop_names='[]'")
        dataset = DropNamesDataset(dataset, cfg.dataset.drop_names)

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

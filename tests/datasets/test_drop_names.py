"""--dataset.drop_names: named state/action entries (e.g. XLeRobot's head servos) hidden from training."""

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.factory import DropNamesDataset

NAMES = ["arm.pos", "head_motor_1.pos", "head_motor_2.pos", "x.vel"]


class FakeDataset:
    def __init__(self):
        vec = ("observation.state", "action")
        features = {k: {"dtype": "float32", "shape": (4,), "names": list(NAMES)} for k in vec}
        features["observation.images.head"] = {"dtype": "video", "shape": (3, 2, 2), "names": ["c", "h", "w"]}
        stats = {k: {"min": np.arange(4.0), "max": np.arange(4.0) + 1, "count": np.array([7])} for k in vec}
        self.meta = SimpleNamespace(features=features, stats=stats)
        self.num_frames = 3

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return {"observation.state": torch.arange(4.0), "action": torch.arange(8.0).reshape(2, 4)}


def test_drop_names_slices_items_features_and_stats():
    ds = DropNamesDataset(FakeDataset(), ["head_motor_1.pos", "head_motor_2.pos", "not_in_dataset.pos"])
    assert ds[0]["observation.state"].tolist() == [0.0, 3.0]
    assert ds[0]["action"].tolist() == [[0.0, 3.0], [4.0, 7.0]]
    assert ds.meta.features["action"]["names"] == ["arm.pos", "x.vel"]
    assert ds.meta.features["action"]["shape"] == (2,)
    assert ds.meta.features["observation.images.head"]["shape"] == (3, 2, 2)
    assert ds.meta.stats["action"]["min"].tolist() == [0.0, 3.0]
    assert ds.meta.stats["action"]["count"].tolist() == [7]
    assert len(ds) == 3 and ds.num_frames == 3
    # DataLoader workers get a pickled copy.
    assert pickle.loads(pickle.dumps(ds))[0]["observation.state"].tolist() == [0.0, 3.0]

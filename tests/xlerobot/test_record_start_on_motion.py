"""record_loop writes nothing until the leader moves, then the last still frame + every frame after it."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rec", Path(__file__).parents[2] / "examples/xlerobot/record_remote_bi_so101_leader_keyboard.py"
)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

KEYS = [f"{s}_arm_{j}.pos" for s in ("left", "right") for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                                 "wrist_flex", "wrist_roll", "gripper")]


class Robot:
    teleop_keys = {"quit": "b"}
    action_features = dict.fromkeys(KEYS, float)

    def get_observation(self):
        return dict.fromkeys(KEYS, 0.0)

    def _from_keyboard_to_base_action(self, keys):
        return {}

    def send_action(self, a):
        return a


class Leader:
    def __init__(self, poses, events):
        self.poses, self.t, self.events = poses, 0, events

    def get_action(self):
        self.t += 1
        if self.t >= len(self.poses):
            self.events["exit_early"] = True  # the operator presses -> after the last pose
        return dict.fromkeys(KEYS, self.poses[min(self.t - 1, len(self.poses) - 1)])


class Keyboard:
    def get_action(self):
        return {}


class Dataset:
    features = {}

    def __init__(self):
        self.frames = []

    def add_frame(self, f):
        self.frames.append(f)


def run(poses, start_tol):
    rec.build_dataset_frame = lambda feats, d, prefix: {prefix: dict(d[0] if isinstance(d, tuple) else d)}
    ds, events = Dataset(), {"exit_early": False}
    rec.busy_wait = lambda s: None
    got = rec.record_loop(Robot(), Leader(poses, events), Keyboard(), events, fps=30, control_time_s=60,
                          dataset=ds, single_task="t", display_data=False, teleop_action_processor=lambda x: x[0],
                          robot_action_processor=lambda x: x[0], robot_observation_processor=lambda o: o,
                          max_arm_step_per_frame=100, start_tol=start_tol)
    return got, ds.frames


def test_waits_for_motion():
    # still for 5 steps (1.0 is under the 2.0 tolerance), then moves
    poses = [0, 0, 1.0, 0, 0] + [5.0] * 20
    got, frames = run(poses, 2.0)
    assert got == len(frames) == 1 + 20, got  # the last still frame + the 20 moving steps
    first, second = (f["action"][KEYS[0]] for f in frames[:2])  # recorded = the smoothed command (alpha 0.8)
    assert abs(first) < 1 and second > 2, (first, second)


def test_start_tol_none_records_everything():
    got, frames = run([0.0] * 10, None)
    assert got == 10


def test_never_moves_writes_nothing():
    got, frames = run([0.0] * 30, 2.0)
    assert got == 0 and frames == []

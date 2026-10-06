"""Regression checks for evaluation keyboard phases, result saving, and resume."""

import csv
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.scripts import lerobot_record_eval as evaluation


class EvaluationTests(unittest.TestCase):
    def test_repeated_enter_waits_for_one_explicit_result(self):
        for answer, expected in (("\x1b[CY", True), ("n", False)):
            with self.subTest(answer=answer), patch(
                "builtins.input", side_effect=["", "", "\x1b[D", "maybe", answer]
            ) as read:
                self.assertEqual(evaluation.ask_success(), expected)
            prompts = [call.args[0] for call in read.call_args_list]
            self.assertEqual(sum("success?" in prompt for prompt in prompts), 1)
            self.assertEqual(prompts[1:4], ["", "", ""])
            self.assertEqual(prompts[4], "Please enter y or n: ")

    def make_listener(self):
        listeners = []

        def create_listener(on_press):
            listener = Mock(on_press=on_press)
            listeners.append(listener)
            return listener

        pynput = ModuleType("pynput")
        pynput.keyboard = SimpleNamespace(
            Key=SimpleNamespace(right="right", left="left", esc="esc"),
            Listener=create_listener,
        )
        with patch.dict("sys.modules", {"pynput": pynput}), patch.object(evaluation, "is_headless", return_value=False):
            listener, events = evaluation.init_eval_keyboard_listener()
        self.assertEqual(listeners, [listener])
        return listener, events

    def test_arrows_are_ignored_while_idle_but_escape_stops(self):
        listener, events = self.make_listener()
        listener.on_press("right")
        listener.on_press("left")
        self.assertFalse(events["exit_early"])
        self.assertFalse(events["rerecord_episode"])
        listener.on_press("esc")
        self.assertTrue(events["stop_recording"])

    def test_arrows_control_an_active_episode(self):
        listener, events = self.make_listener()
        events["recording_active"] = True
        listener.on_press("right")
        self.assertTrue(events["exit_early"])
        events["exit_early"] = False
        listener.on_press("left")
        self.assertTrue(events["exit_early"])
        self.assertTrue(events["rerecord_episode"])

    def test_failed_episode_save_does_not_write_a_success_label(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = SimpleNamespace(
                root=root, num_episodes=2, episode_buffer={"size": 1},
                save_episode=Mock(side_effect=RuntimeError("encoding failed")),
            )
            with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                evaluation.save_evaluation_episode(dataset, True, "checkpoint")
            self.assertFalse((Path(root) / "success.csv").exists())

    def test_empty_episode_cannot_receive_a_success_label(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = SimpleNamespace(
                root=root, num_episodes=2, episode_buffer={"size": 0}, save_episode=Mock(),
            )
            with self.assertRaisesRegex(ValueError, "without frames"):
                evaluation.save_evaluation_episode(dataset, True, "checkpoint")
            dataset.save_episode.assert_not_called()
            self.assertFalse((Path(root) / "success.csv").exists())

    def test_history_excludes_labels_for_unsaved_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "success.csv"
            path.write_text("episode,success,policy\n0,0,checkpoint\n1,1,checkpoint\n2,1,checkpoint\n")
            self.assertEqual(evaluation.load_success_history(path, 2), [False, True])

    def test_resume_and_keys_during_encoding_without_robot_or_cameras(self):
        for frame_counts in ([1, 1], [0, 1, 1]):
            with self.subTest(frame_counts=frame_counts), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                checkpoint = root / "checkpoint"
                checkpoint.mkdir()
                (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "test/training"}}))
                (root / "success.csv").write_text("episode,success,policy\n0,0,checkpoint\n1,1,checkpoint\n")
                listener, events = self.make_listener()
                dataset = SimpleNamespace(root=root, num_episodes=2, meta=SimpleNamespace(stats={}))
                dataset.create_episode_buffer = lambda: {"size": 0, "episode_index": dataset.num_episodes}
                dataset.start_image_writer = Mock()
                dataset.finalize = Mock()

                def save_episode():
                    self.assertFalse(events["recording_active"])
                    # Reproduce the user's arrow press during video encoding.
                    listener.on_press("right")
                    self.assertFalse(events["exit_early"])
                    dataset.num_episodes += 1
                    dataset.episode_buffer = dataset.create_episode_buffer()

                dataset.save_episode = save_episode
                robot = SimpleNamespace(
                    is_connected=False, action_features={}, observation_features={},
                    config=SimpleNamespace(cameras={"head": None}),
                )
                robot.connect = lambda: setattr(robot, "is_connected", True)
                robot.disconnect = lambda: setattr(robot, "is_connected", False)
                cfg = SimpleNamespace(
                    robot=robot.config, teleop=None,
                    policy=SimpleNamespace(pretrained_path=str(checkpoint), device="cpu"),
                    dataset=SimpleNamespace(
                        repo_id="test/eval", root=root, fps=30, video=True, rename_map={}, video_encoding_batch_size=1,
                        num_image_writer_processes=0, num_image_writer_threads_per_camera=4,
                        num_episodes=4, episode_time_s=60, single_task="handover", push_to_hub=False,
                    ),
                    resume=True, display_data=False, play_sounds=False,
                    reset_dataset=None, reset_speed=20, max_arm_step=3, allow_base=False,
                )
                counts = iter(frame_counts)

                def record_loop(**kwargs):
                    self.assertTrue(events["recording_active"])
                    self.assertFalse(events["exit_early"])
                    dataset.episode_buffer["size"] = next(counts)

                replacements = {
                    "init_logging": Mock(), "asdict": lambda cfg: {},
                    "make_robot_from_config": lambda cfg: robot,
                    "make_default_processors": lambda: (None, None, None),
                    "create_initial_features": Mock(), "aggregate_pipeline_dataset_features": Mock(return_value={}),
                    "LeRobotDataset": Mock(return_value=dataset),
                    "sanity_check_dataset_robot_compatibility": Mock(),
                    "make_policy": Mock(return_value=object()), "make_pre_post_processors": Mock(return_value=(None, None)),
                    "init_eval_keyboard_listener": lambda: (listener, events),
                    "dataset_start_pose": Mock(return_value={}), "reset_arms": Mock(),
                    "VideoEncodingManager": lambda ds: nullcontext(), "record_loop": record_loop,
                    "log_say": Mock(), "is_headless": lambda: False,
                }
                with ExitStack() as stack:
                    for name, value in replacements.items():
                        stack.enter_context(patch.object(evaluation, name, value))
                    prompts = stack.enter_context(patch("builtins.input", return_value=""))
                    success = stack.enter_context(patch.object(evaluation, "ask_success", return_value=True))
                    evaluation.record.__wrapped__(cfg)
                self.assertEqual(dataset.num_episodes, 4)
                self.assertEqual(success.call_count, 2)
                self.assertIn("trial 3 of 4", prompts.call_args_list[0].args[0])
                dataset.start_image_writer.assert_called_once()
                self.assertFalse(robot.is_connected)
                dataset.finalize.assert_called_once()
                with (root / "success.csv").open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([int(row["episode"]) for row in rows], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

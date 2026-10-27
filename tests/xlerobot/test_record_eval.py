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
            Key=SimpleNamespace(right="right", left="left", esc="esc", space="space"),
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
                    reset_dataset=None, reset_speed=20, max_arm_step=3, allow_base=False, takeover=False,
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

    def test_smoothing_cancels_zigzag_and_limits_steps(self):
        k = "left_arm_elbow_flex.pos"
        prev, hist, sent = {k: 10.0}, [], []
        for t in range(40):  # policy output zig-zags 10 +- 1 every frame
            action, hist = evaluation.smooth_arm_action({k: 10.0 + (-1) ** t, "x.vel": 0.3}, hist, prev, 3.0)
            prev = action
            sent.append(action[k])
        self.assertLess(max(sent[10:]) - min(sent[10:]), 0.1)
        self.assertEqual(action["x.vel"], 0.3)
        action, _ = evaluation.smooth_arm_action({k: 90.0}, [{k: 90.0}, {k: 90.0}], {k: 10.0}, 3.0)
        self.assertEqual(action[k], 13.0)

    def test_takeover_aligns_leader_then_records_its_span(self):
        k = "left_arm_elbow_flex.pos"

        class Bus:  # leader arm servo bus: position follows the last goal while torque is on
            motors = ["elbow_flex"]

            def __init__(self, silent=False):
                self.pos, self.torque, self.silent, self.limits = {"elbow_flex": 80.0}, False, silent, []

            def sync_read(self, *_, **__):
                return dict(self.pos)

            def sync_write(self, name, values, **__):
                if name == "Torque_Limit":
                    self.limits.append(values["elbow_flex"])
                elif self.torque:
                    self.pos.update(values)

            def enable_torque(self, *_, **__):
                if self.silent:  # a motor that does not answer, as seen on the real leader
                    raise ConnectionError("There is no status packet!")
                self.torque = True

            def disable_torque(self, *_, **__):
                self.torque = False

        teleop = SimpleNamespace(left_arm=SimpleNamespace(bus=Bus()), right_arm=SimpleNamespace(bus=Bus()))
        g = "left_arm_gripper.pos"
        robot = SimpleNamespace(pos=10.0, grip=40.0, action_features={}, robot_type="xlerobot_client")
        robot.get_observation = lambda: {k: robot.pos, g: robot.grip}
        grips_sent = []

        def send(a):
            robot.pos, robot.grip = a[k], a[g]
            grips_sent.append(a[g])
            return a

        sent = []
        robot.send_action = lambda a: (sent.append(a[k]), send(a))[1]
        dataset = SimpleNamespace(fps=30, features={}, episode_buffer={"size": 0})
        dataset.add_frame = lambda _: dataset.episode_buffer.update(size=dataset.episode_buffer["size"] + 1)
        listener, events = self.make_listener()
        events["recording_active"] = True
        frames, pushes, squeeze = iter(range(1000)), iter(range(0, 5000, 5)), iter(range(60, -5000, -15))
        script = {3: "space", 9: "space", 14: "right"}  # takeover after frame 3, back after 9, end at 14

        def tick():  # one control frame, whoever drives it
            i = next(frames)
            if i in script:
                listener.on_press(script[i])
            return i

        def policy_step(**_):
            # Before the takeover the policy holds still; after the hand-back it wants to be 50 units away at once.
            return {k: robot.pos + (50.0 if tick() > 9 else 0.0), g: robot.grip}

        def leader_step():
            tick()  # the operator moves the leader 5 units further every frame
            # ...and squeezes the leader gripper from wide open (60) past the held grip (40) to closed
            return {k: teleop.left_arm.bus.pos["elbow_flex"] + next(pushes), g: float(next(squeeze))}

        teleop.get_action = leader_step

        with patch.object(evaluation, "predict_action", side_effect=policy_step), patch.object(
            evaluation, "make_robot_action", side_effect=lambda a, _: a
        ), patch.object(evaluation, "build_dataset_frame", return_value={}), patch.object(
            evaluation, "precise_sleep"
        ), patch.object(evaluation.time, "sleep"):
            spans = evaluation.record_loop(
                robot=robot, events=events, fps=30, dataset=dataset, teleop=teleop,
                policy=Mock(config=SimpleNamespace(device="cpu", use_amp=False)), preprocessor=Mock(steps=[]), postprocessor=Mock(), control_time_s=60,
                teleop_action_processor=lambda x: x[0], robot_action_processor=lambda x: x[0],
                robot_observation_processor=lambda o: o, max_arm_step=3.0, takeover=True, leader_rest={k: -50.0},
            )
        # Takeover: the leader met the follower pose (10), its torque faded out (slowly first, then faster) and
        # came back to full strength once released. Hand-back: it went to the rest pose (-50), released again.
        bus = teleop.left_arm.bus
        self.assertEqual(bus.pos["elbow_flex"], -50.0)
        self.assertFalse(bus.torque)
        fade = bus.limits[: bus.limits.index(1000, 1)]
        self.assertEqual((fade[0], fade[-1]), (1000, 0))
        drops = [a - b for a, b in zip(fade, fade[1:])]
        self.assertTrue(all(d >= 0 for d in drops) and drops[-1] > 5 * max(drops[0], 1))
        self.assertEqual(bus.limits[len(fade)], 1000)
        self.assertEqual(bus.limits[-1], 1000)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(start, 4)  # policy frames 0-3 come first
        self.assertGreater(end - start, 0)
        self.assertGreater(robot.pos, 10.0)  # the leader moved the robot, at most 3 units per frame
        self.assertLessEqual(robot.pos, 10.0 + 3.0 * (end - start))
        # After each switch the command eases (smoothstep, 1 s) from the last command sent: the grip from the
        # policy's 40 towards the leader's absolute value (45, 30, 15, ...; no step limit), and the arm after
        # the hand-back towards the policy's far target. Neither jumps on the switch frame.
        ease = lambda n: (lambda x: x * x * (3 - 2 * x))(min(1.0, n / 30))
        expected = [40.0 + ease(i + 1) * (45.0 - 15 * i - 40.0) for i in range(end - start)]
        for got, want in zip(grips_sent[start:end], expected):
            self.assertAlmostEqual(got, want)
        steps = [abs(b - a) for a, b in zip(sent, sent[1:])]
        self.assertLess(steps[end - 1], 0.1)  # first policy command after the hand-back
        self.assertLess(steps[start - 1], 0.1)  # first leader command after the takeover
        with tempfile.TemporaryDirectory() as root:
            evaluation.save_interventions(Path(root), 7, spans)
            evaluation.save_interventions(Path(root), 8, [])
            evaluation.save_interventions(Path(root), 9, [(30, 30)])  # space pressed as the episode ended
            with (Path(root) / "interventions.csv").open(newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [["episode", "start_frame", "end_frame"], ["7", str(start), str(end)]])

        # A silent leader motor skips the alignment instead of crashing; torque is released everywhere.
        teleop.right_arm.bus = Bus(silent=True)
        evaluation.align_leader(teleop, {k: 10.0, "right_arm_elbow_flex.pos": 10.0}, 30)
        self.assertFalse(teleop.left_arm.bus.torque or teleop.right_arm.bus.torque)

        # Joints calibrated inside the encoder range are aligned (grippers too); one across the wrap is never driven.
        bus = Bus()
        bus.motors, bus.pos = ["elbow_flex", "gripper"], {"elbow_flex": 80.0, "gripper": 50.0}
        bus.calibration = {"elbow_flex": SimpleNamespace(range_min=900, range_max=3100),
                           "gripper": SimpleNamespace(range_min=6088, range_max=7298)}
        teleop.left_arm.bus, teleop.right_arm.bus = bus, Bus()
        with patch.object(evaluation.time, "sleep"):
            evaluation.align_leader(teleop, {k: 10.0, "left_arm_gripper.pos": 5.0}, 30, stop=lambda: True)
        self.assertEqual(bus.pos, {"elbow_flex": 10.0, "gripper": 50.0})
        bus.calibration["gripper"] = SimpleNamespace(range_min=2031, range_max=3243)
        with patch.object(evaluation.time, "sleep"):
            evaluation.align_leader(teleop, {k: 10.0, "left_arm_gripper.pos": 5.0}, 30, stop=lambda: True)
        self.assertEqual(bus.pos, {"elbow_flex": 10.0, "gripper": 5.0})

        # The episode stopping cuts the fade short; the leader is still released at full torque limit.
        teleop.left_arm.bus, teleop.right_arm.bus = Bus(), Bus()
        stops = iter([False] * 5 + [True])
        with patch.object(evaluation.time, "sleep"):
            evaluation.align_leader(teleop, {k: 10.0}, 30, stop=lambda: next(stops), show=False)
        self.assertEqual(teleop.left_arm.bus.pos["elbow_flex"], 10.0)
        self.assertEqual(len(teleop.left_arm.bus.limits), 6)  # 5 fade steps, then the restore
        self.assertEqual(teleop.left_arm.bus.limits[-1], 1000)
        self.assertFalse(teleop.left_arm.bus.torque or teleop.right_arm.bus.torque)

    def test_reset_arms_moves_gripper_half_open_at_double_speed_with_the_arm(self):
        g, j = "left_arm_gripper.pos", "left_arm_elbow_flex.pos"
        sent = []
        robot = SimpleNamespace(get_observation=lambda: {g: 10.0, j: 0.0}, action_features={"x.vel": float},
                                send_action=sent.append)
        with patch.object(evaluation.time, "sleep"):
            evaluation.reset_arms(robot, {g: 20.0, j: 120.0}, 30, 40.0)
        # Arm: 120 units at 40/s = 3 s; gripper: 40 + 30 units at 80/s < 1 s, both start on the first frame.
        self.assertEqual(len(sent), 3 * 30 + 1)
        self.assertGreater(sent[1][g], 10.0)
        self.assertGreater(sent[1][j], 0.0)
        self.assertAlmostEqual(max(a[g] for a in sent), 50.0, delta=0.5)
        done = next(i for i, a in enumerate(sent) if a[g] == 20.0)
        self.assertLess(done, 30)
        self.assertEqual(sent[-1], {g: 20.0, j: 120.0, "x.vel": 0.0})


if __name__ == "__main__":
    unittest.main()

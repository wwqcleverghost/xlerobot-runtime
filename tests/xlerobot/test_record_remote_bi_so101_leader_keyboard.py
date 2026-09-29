"""Hardware-free checks for the teleoperation safety limit and diagnostics."""

import unittest

from examples.xlerobot.record_remote_bi_so101_leader_keyboard import (
    ControlLoopMonitor,
    smooth_arm_action,
)


class TestTeleopSmoothing(unittest.TestCase):
    def test_velocity_limit_uses_elapsed_time_and_preserves_base_command(self):
        action = {"left_arm_elbow_flex.pos": 20.0, "right_arm_elbow_flex.pos": -20.0, "x.vel": 0.4}
        previous = {"left_arm_elbow_flex.pos": 0.0, "right_arm_elbow_flex.pos": 0.0}

        fast_loop = smooth_arm_action(action, previous, dt_s=0.02, alpha=0.8, max_velocity=90.0)
        slow_loop = smooth_arm_action(action, previous, dt_s=0.05, alpha=0.8, max_velocity=90.0)

        self.assertAlmostEqual(fast_loop["left_arm_elbow_flex.pos"], 1.8)
        self.assertAlmostEqual(fast_loop["right_arm_elbow_flex.pos"], -1.8)
        self.assertAlmostEqual(slow_loop["left_arm_elbow_flex.pos"], 4.5)
        self.assertEqual(fast_loop["x.vel"], action["x.vel"])

    def test_ema_acts_without_hitting_velocity_limit(self):
        result = smooth_arm_action(
            {"left_arm_wrist_roll.pos": 10.0},
            {"left_arm_wrist_roll.pos": 0.0},
            dt_s=0.1,
            alpha=0.5,
            max_velocity=100.0,
        )
        self.assertAlmostEqual(result["left_arm_wrist_roll.pos"], 5.0)

    def test_invalid_elapsed_time_cannot_bypass_limit(self):
        for dt_s in (0.0, -0.1, float("nan")):
            with self.subTest(dt_s=dt_s), self.assertRaisesRegex(ValueError, "dt_s"):
                smooth_arm_action({"left_arm_wrist_roll.pos": 10.0}, {"left_arm_wrist_roll.pos": 0.0}, dt_s)

    def test_monitor_reports_timing_and_follower_error(self):
        monitor = ControlLoopMonitor(target_fps=30)
        monitor.record_tracking_error(
            {"left_arm_elbow_flex.pos": 8.0, "right_arm_elbow_flex.pos": -5.0},
            {"left_arm_elbow_flex.pos": 10.0, "right_arm_elbow_flex.pos": -10.0},
        )
        monitor.record_timing(loop_period_s=0.04, send_interval_s=0.03, work_s=0.02, capped=True)

        with self.assertLogs(level="INFO") as logs:
            monitor.report_if_due(monitor.window_start + 1.0)

        message = "\n".join(logs.output)
        self.assertIn("actual FPS 1.0/target 30", message)
        self.assertIn("loop dt mean/max 40.0/40.0 ms", message)
        self.assertIn("stalled dt caps 1", message)
        self.assertIn("follower error mean/max 3.50/5.00", message)
        self.assertIn("left max 2.00, right max 5.00", message)


if __name__ == "__main__":
    unittest.main()

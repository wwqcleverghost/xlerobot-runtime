"""Forward kinematics of the two XLeRobot arms from recorded joint values (observation.state / action, first 12 dims).

Arm chain: so101_new_calib.urdf (TheRobotStudio/SO-ARM100, Simulation/SO101, commit 385e8d7c68, 2025-07-02), whose joint zero is the
middle of each joint's range, like LeRobot's calibration. Recorded values are RANGE_M100_100 (use_degrees=False in config_xlerobot.py):
0 = middle of the calibrated range, +-100 = its ends, so angle = value / 100 x half the range in ticks x 2pi / 4096.
Positions are in the right arm's base frame (x forward, y left, z up); the left arm base sits at LEFT_BASE.
Checked on v7-handover-clean (2026-11-02): just before A lets go (both grippers on the tape) the two tips are a median 4.9 cm
apart with LEFT_BASE fitted to (-1.9, 24.3, 0.3) cm (xlerobot.urdf says 26.6 cm apart, same height and heading); flipping
shoulder_lift / elbow_flex / wrist_flex gives 18 cm, so their signs are right.
"""
import numpy as np

# so101_new_calib.urdf: (xyz, rpy) of each revolute joint (all about z), base -> wrist_roll, then the fixed gripper_frame_joint
CHAIN = [((0.0388353, 0, 0.0624), (3.14159, 0, -3.14159)),          # shoulder_pan
         ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0)),  # shoulder_lift
         ((-0.11257, -0.028, 0), (0, 0, 1.5708)),                     # elbow_flex
         ((-0.1349, 0.0052, 0), (0, 0, -1.5708)),                     # wrist_flex
         ((0, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159))]        # wrist_roll
TIP = ((-0.0079, -0.000218121, -0.0981274), (0, 3.14159, 0))           # gripper_frame_link: between the jaw tips
# calibration xlr26009 (lab PC ~/.cache/huggingface/lerobot/calibration/robots/xlerobot/, 2026-08-20): (range_min, range_max) ticks
CAL = {"L": [(828, 3566), (856, 3238), (946, 3148), (875, 3220), (136, 3958)],
       "R": [(827, 3567), (851, 3243), (944, 3151), (874, 3220), (139, 3955)]}
LEFT_BASE = np.array([-0.019, 0.243, 0.003])   # m, fitted on handovers (see above)
SLICE = {"L": slice(0, 5), "R": slice(6, 11)}


def _rot(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy, q=0.0):
    T = np.eye(4)
    T[:3, :3] = _rot(*rpy) @ _rot(0, 0, q)
    T[:3, 3] = xyz
    return T


def to_rad(values, arm):
    """5 recorded joint values (RANGE_M100_100) of one arm -> radians from the middle of each range."""
    return np.array([v / 100 * (hi - lo) / 2 * 2 * np.pi / 4096 for v, (lo, hi) in zip(values, CAL[arm])])


def arm_points(state, arm):
    """(wrist, tip) positions in the right-arm base frame for one frame; wrist = the wrist_roll joint (gripper_link origin)."""
    T = np.eye(4)
    for (xyz, rpy), q in zip(CHAIN, to_rad(np.asarray(state)[SLICE[arm]], arm)):
        T = T @ _tf(xyz, rpy, q)
    off = LEFT_BASE if arm == "L" else 0
    return T[:3, 3] + off, (T @ _tf(*TIP))[:3, 3] + off


def view_angle(state, A, B):
    """Degrees between A's gripper axis (wrist -> tip, roughly where its wrist camera looks) and the line from A's wrist to B's tip."""
    wa, ta = arm_points(state, A)
    _, tb = arm_points(state, B)
    ax, v = ta - wa, tb - wa
    return float(np.degrees(np.arccos(np.clip(ax @ v / np.linalg.norm(ax) / np.linalg.norm(v), -1, 1))))


def selfcheck():
    # rest pose of v7 ep0 (folded arms): tips ~15 cm in front of each base, near base height, arms ~24 cm apart
    s = np.zeros(12)
    s[0:5], s[6:11] = [3, -100, 95, 73, 4], [-3, -99, 96, 74, -1]
    (_, tl), (_, tr) = arm_points(s, "L"), arm_points(s, "R")
    assert 0.12 < tr[0] < 0.18 and abs(tr[1]) < 0.03 and abs(tr[2]) < 0.03, tr
    assert abs(tl[1] - tr[1] - 0.243) < 0.03, (tl, tr)


if __name__ == "__main__":
    selfcheck()
    print("ok")

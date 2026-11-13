"""Stage progress feedback in the diffusion policy (config.progress_mode / progress_check_*)."""
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make(**kw):
    cfg = DiffusionConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (12,)), OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (9,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (21,))},
        horizon=16, n_action_steps=8, noise_scheduler_type="FLOW", down_dims=(32, 64), **kw)
    return DiffusionModel(cfg)


def batch(v=3.4):
    s = DiffusionModel.cumulative(torch.full((4, 2), v), 9)
    a = torch.cat([torch.zeros(4, 16, 12), DiffusionModel.cumulative(torch.linspace(v, v + 0.5, 16).expand(4, 16), 9)], -1)
    return {OBS_STATE: torch.zeros(4, 2, 12), OBS_ENV_STATE: s, ACTION: a, "action_is_pad": torch.zeros(4, 16, dtype=torch.bool)}


def test_targets_and_losses():
    assert torch.allclose(DiffusionModel.cumulative(torch.tensor(3.4), 9), torch.tensor([1, 1, 1, .4, 0, 0, 0, 0, 0]))
    b = batch()
    for mode in ("self", "delta", "deltainc"):
        m = make(progress_mode=mode, progress_check_weight=1.0)
        nb, true_s = m._progress_batch(b)
        assert torch.equal(true_s, b[OBS_ENV_STATE])
        ref = {"self": 0, "delta": nb[OBS_ENV_STATE][:, -1:], "deltainc": true_s[:, -1:]}[mode]
        assert torch.allclose(nb[ACTION][..., 12:], b[ACTION][..., 12:] - ref)
        assert torch.equal(nb[ACTION][..., :12], b[ACTION][..., :12])
        loss = m.compute_loss(b)
        assert loss.isfinite() and "progress_check_mse" in m.stage_metrics
    m = make(progress_mode="self", progress_check_weight=1.0, progress_check_only=True)
    assert m.compute_loss(b).isfinite()


def test_eval_check():
    p = DiffusionPolicy(make(progress_mode="self", progress_check_weight=1.0).config)
    s2 = p.progress_check({OBS_STATE: torch.zeros(4, 12)}, torch.zeros(4, 9), torch.zeros(4, 9))
    assert s2.shape == (4, 9) and ((s2 >= 0) & (s2 <= 1)).all()


def test_eval_memory():
    """Eval feeds the policy its own progress: zeros first, updated once per chunk (8 actions) from the step after it."""
    import numpy as np

    from lerobot.scripts.lerobot_record_eval import ProgressMemory, live_stage

    for mode in ("self", "delta"):
        p = DiffusionPolicy(make(progress_mode=mode).config)
        p.reset()
        mem = live_stage(p)
        assert isinstance(mem, ProgressMemory) and not mem().any()
        seen = []
        for i in range(16):
            before = mem().copy()
            a = p.select_action({OBS_STATE: torch.zeros(1, 12), OBS_ENV_STATE: torch.from_numpy(mem())[None]})
            assert a.shape == (1, 21)
            stage, prog = mem.update(p)
            nxt = p.diffusion.progress_next[0].numpy()  # chunk step 9 = 8 frames after the planning observation
            want = np.clip(nxt if mode == "self" else before + nxt, 0, 1)
            assert np.allclose(mem(), want if i % 8 == 7 else before)
            assert 0 <= stage <= 8 and 0 <= prog <= 1
            seen.append(not np.array_equal(mem(), before))
        assert all(not s for i, s in enumerate(seen) if i % 8 != 7)


def test_eval_memory_deltainc():
    """deltainc adds the total increment to v = sum(s) and keeps s a cumulative array: noise on a finished stage cannot sink it."""
    import numpy as np

    from lerobot.scripts.lerobot_record_eval import live_stage

    p = DiffusionPolicy(make(progress_mode="deltainc").config)
    p.reset()
    mem = live_stage(p)
    mem.s = np.array([1, 1, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    p.diffusion.progress_next = torch.tensor([[-0.2, 0.1, 0.3, 0.05, 0, 0, 0, 0, 0]])  # sum 0.25
    mem.update(p)  # the action queue is empty after reset = a chunk just ended
    assert np.allclose(mem(), [1, 1, 0.75, 0, 0, 0, 0, 0, 0])


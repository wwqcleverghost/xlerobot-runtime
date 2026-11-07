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

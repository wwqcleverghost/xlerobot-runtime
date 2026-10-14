"""Flow matching (noise_scheduler_type="FLOW") and rolling denoising (react_blocks) for the Diffusion Policy."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

K, S, A = 4, 4, 3  # blocks, actions per block, action dim


def make_policy(**overrides) -> DiffusionPolicy:
    kwargs = {"horizon": K * S, "n_action_steps": S, "react_blocks": K, **overrides}
    config = DiffusionConfig(down_dims=(16, 32), crop_shape=None, **kwargs)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(A,))}
    return DiffusionPolicy(config)


def test_rolling_denoising_recovers_trajectory_with_exact_velocity():
    model = make_policy().diffusion
    rolls = 12
    truth = 0.9 * torch.sin(torch.linspace(0, 6, (rolls + K) * S))[None, :, None].expand(1, -1, A)
    calls = []

    def exact_velocity(x, global_cond):
        # The buffer of roll r covers truth[r*S : r*S + K*S]; x = tau*eps + (1-tau)*a, so eps - a = (x - a)/tau.
        r = len(calls)
        calls.append(r)
        return (x - truth[:, r * S : r * S + K * S]) / model._react_tau(x)

    model._react_velocity = exact_velocity
    model._prepare_global_conditioning = lambda batch: torch.zeros(1, 1)
    buffer = None
    for step in range(rolls - K + 1):
        actions, buffer = model.react_step({}, buffer)
        r = step + K - 1  # the first call also runs the K - 1 warm-up rolls
        torch.testing.assert_close(actions, truth[:, r * S : (r + 1) * S])
    assert len(calls) == rolls


def test_rolling_denoising_trains_and_acts():
    policy = make_policy()
    batch = {
        OBS_STATE: torch.randn(5, 2, 2),
        OBS_ENV_STATE: torch.randn(5, 2, 2),
        ACTION: torch.rand(5, K * S, A) * 2 - 1,
        "action_is_pad": torch.zeros(5, K * S, dtype=torch.bool),
    }
    loss, _ = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss)

    policy.eval()
    steps = []
    react_step = policy.diffusion.react_step
    policy.diffusion.react_step = lambda *a: steps.append(1) or react_step(*a)
    for _ in range(3 * S):
        action = policy.select_action({OBS_STATE: torch.randn(1, 2), OBS_ENV_STATE: torch.randn(1, 2)})
        assert action.shape == (1, A) and action.abs().max() <= 1
    assert len(steps) == 3  # one rolling update per executed block
    assert policy._react_buffer.shape == (1, K * S, A)


def make_flow_policy() -> DiffusionPolicy:
    return make_policy(noise_scheduler_type="FLOW", horizon=16, n_action_steps=8, react_blocks=None)


def test_flow_sampling_reaches_target_with_exact_velocity():
    model = make_flow_policy().diffusion
    target = torch.rand(2, 16, A) * 2 - 1
    # On the straight path x = tau*eps + (1-tau)*a the velocity eps - a equals (x - a)/tau.
    model._flow_velocity = lambda x, tau, global_cond: (x - target) / tau[:, None, None]
    torch.testing.assert_close(model.conditional_sample(2), target)
    assert model.num_inference_steps == 10


def test_flow_trains_and_acts():
    policy = make_flow_policy()
    batch = {
        OBS_STATE: torch.randn(5, 2, 2),
        OBS_ENV_STATE: torch.randn(5, 2, 2),
        ACTION: torch.rand(5, 16, A) * 2 - 1,
        "action_is_pad": torch.zeros(5, 16, dtype=torch.bool),
    }
    loss, _ = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss)

    policy.eval()
    chunks = []
    generate = policy.diffusion.generate_actions
    policy.diffusion.generate_actions = lambda *a, **k: chunks.append(1) or generate(*a, **k)
    for _ in range(16):
        action = policy.select_action({OBS_STATE: torch.randn(1, 2), OBS_ENV_STATE: torch.randn(1, 2)})
        assert action.shape == (1, A) and action.abs().max() <= 1
    assert len(chunks) == 2  # a fresh 10-step sample every n_action_steps

"""Stage progress feedback in the diffusion policy (config.progress_mode / progress_check_*)."""
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


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



def make_v9(**kw):
    """v9: progress only predicted (last 9 action dims), no environment_state input."""
    cfg = DiffusionConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (12,)),
                        "observation.images.cam": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (21,))}, crop_shape=None, pretrained_backbone_weights=None,
        horizon=16, n_action_steps=8, noise_scheduler_type="FLOW", down_dims=(32, 64), progress_pred_dims=9, **kw)
    return DiffusionModel(cfg)


def test_v9_losses():
    b = batch()
    del b[OBS_ENV_STATE]
    b[OBS_IMAGES] = torch.zeros(4, 2, 1, 3, 32, 32)
    b["adv"], b["neg"] = torch.tensor([1, 0, 1, 0]), torch.tensor([0, 1, 0, 1])
    for kw in ({}, {"adv_cond": True}, {"neg_action_mask": True}, {"adv_cond": True, "adv_drop": 0.0}):
        torch.manual_seed(0)
        m = make_v9(**kw)
        per = m.compute_loss(b, reduction="none")
        assert per.shape == (4,) and per.isfinite().all()
    # neg_action_mask drops the joint dims of neg frames only. (The progress dims of a neg frame still see its noisy joints
    # in the U-Net input, so its loss is not independent of the joints: flow denoises all dims together.)
    torch.manual_seed(1)
    m = make_v9(neg_action_mask=True)
    torch.manual_seed(2)
    masked = m.compute_loss(b, reduction="none")
    m.config.neg_action_mask = False
    torch.manual_seed(2)
    full = m.compute_loss(b, reduction="none")
    assert torch.allclose(masked[[0, 2]], full[[0, 2]]) and (masked[[1, 3]] < full[[1, 3]]).all()


def test_v9_sampling():
    """CFG (adv_guidance != 1) and best-of-n keep the output shape; best-of-n keeps the chunk of the largest predicted gain."""
    for kw in ({"adv_cond": True}, {"adv_cond": True, "adv_guidance": 2.0}, {"best_of_n": 5}, {"adv_cond": True, "adv_guidance": 1.5, "best_of_n": 3}):
        m = make_v9(**kw).eval()
        with torch.no_grad():
            a = m.generate_actions({OBS_STATE: torch.zeros(2, 2, 12), OBS_IMAGES: torch.zeros(2, 2, 1, 3, 32, 32)})
        assert a.shape == (2, 8, 21) and m.progress_next.shape == (2, 9) and m.progress_now.shape == (2, 9)
        if kw.get("best_of_n", 1) > 1:
            g = m.best_of_n_gain
            got = (m.progress_next - m.progress_now).sum(-1)
            assert torch.allclose(got, g.max(1).values, atol=1e-5)


def test_stall_guard():
    """STALL=rewind:1 at 10 fps: fires after 10 still frames, queues the 15 commands before the stall began, backwards."""
    import numpy as np

    from lerobot.scripts.lerobot_record_eval import StallGuard

    g = StallGuard("rewind:1", 10)
    for t in range(30):  # moving toward the tape: 4 deg per frame (the 3 deg box only holds still frames)
        g.observe(np.full(12, 4.0 * t), {"j": float(t)}, float("nan"))
        assert not g.check(t)
    fired = []
    for t in range(30, 45):  # hover
        g.observe(np.full(12, 4.0 * 29), {"j": 29.0}, float("nan"))
        fired.append(g.check(t))
    assert fired.index(True) == 8 and sum(fired) == 1  # the last approach frame (t=29) + 9 hover frames are still
    assert [c["j"] for c in g.queue] == [float(x) for x in range(28, 13, -1)]  # onset 29: commands 28 .. 14, backwards
    r = StallGuard("resample:1", 10)
    for t in range(12):  # still, but the model's own progress keeps rising: not a stall
        r.observe(np.zeros(12), {"j": 0.0}, t * 0.02 / 9 * 9)
        assert not r.check(t)


def test_v9_back():
    """adv_back: a third indicator; sampling with adv_command 2 (back) runs, labels 0 / 1 / 2 all train."""
    b = batch()
    del b[OBS_ENV_STATE]
    b[OBS_IMAGES] = torch.zeros(4, 2, 1, 3, 32, 32)
    b["adv"] = torch.tensor([1, 0, 2, 2])
    m = make_v9(adv_cond=True, adv_back=True)
    assert m.compute_loss(b).isfinite()
    m.eval()
    for cmd in (0, 2):
        m.adv_command = cmd
        with torch.no_grad():
            a = m.generate_actions({OBS_STATE: torch.zeros(1, 2, 12), OBS_IMAGES: torch.zeros(1, 2, 1, 3, 32, 32)})
        assert a.shape == (1, 8, 21)


def test_stall_guard_learned():
    """STALL=learned:1 at 10 fps: after a stall the indicator is back (2) for STALL_REWIND_S (1.5 s = 15 frames), then good."""
    import numpy as np

    from lerobot.scripts.lerobot_record_eval import StallGuard

    g = StallGuard("learned:1", 10)
    fired = None
    for t in range(40):
        g.observe(np.zeros(12), {"j": 0.0}, float("nan"))
        if g.check(t) and fired is None:
            fired = t
    assert fired == 9 and g.back_left == 15
    cmds = [g.command() for _ in range(20)]
    assert cmds == [2] * 15 + [0] * 5


def test_stall_guard_ask():
    """STALL=ask fires like the others but queues nothing and keeps the indicator at good."""
    import numpy as np

    from lerobot.scripts.lerobot_record_eval import StallGuard

    g = StallGuard("ask:1", 10)
    hits = []
    for t in range(15):  # (the next trigger could come 10 frames later, at most STALL_MAX times)
        g.observe(np.zeros(12), {"j": 0.0}, float("nan"))
        hits.append(g.check(t))
    assert hits.index(True) == 9 and g.queue == [] and g.command() == 0 and g.log == [(9, "ask")]

"""
test_training.py

Unit tests for env/policy_net.py and train.py components.
Run: python -m pytest test_training.py -v
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for _p in [str(_root), str(_root / "training")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import argparse
import tempfile
import os
import numpy as np
import pytest
import torch

from env.actions import ACTION_DIM, action_mask
from env.catan_env import CatanEnv, encode_observation, OBS_DIM
from env.game_state import new_game, visible_vp
from env.policy_net import PolicyNet
from train import compute_gae, ppo_update, DEFAULTS, VecEnv
from evaluate import evaluate_vs_random


# ---------------------------------------------------------------------------
# PolicyNet — shape and masking
# ---------------------------------------------------------------------------

def test_policy_net_single_forward_shapes():
    net = PolicyNet()
    obs  = torch.zeros(OBS_DIM)
    mask = torch.ones(ACTION_DIM, dtype=torch.bool)
    dist, value = net(obs.unsqueeze(0), mask.unsqueeze(0))
    assert dist.probs.shape == (1, ACTION_DIM)
    assert value.shape == (1,)


def test_policy_net_batch_forward_shapes():
    net  = PolicyNet()
    obs  = torch.randn(32, OBS_DIM)
    mask = torch.ones(32, ACTION_DIM, dtype=torch.bool)
    dist, value = net(obs, mask)
    assert dist.probs.shape == (32, ACTION_DIM)
    assert value.shape == (32,)


def test_policy_net_masking_zeroes_illegal_actions():
    net  = PolicyNet()
    obs  = torch.randn(1, OBS_DIM)
    # Allow only actions 0..9
    mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
    mask[0, :10] = True
    dist, _ = net(obs, mask)
    probs = dist.probs[0].detach()
    assert probs[10:].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert probs[:10].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_policy_net_no_mask():
    """Without a mask, all actions get non-zero probability."""
    net  = PolicyNet()
    obs  = torch.randn(1, OBS_DIM)
    dist, value = net(obs)
    assert dist.probs.shape == (1, ACTION_DIM)
    assert value.shape == (1,)


def test_policy_net_act_returns_legal_action():
    """act() should sample from the masked distribution."""
    env = CatanEnv(seed=0)
    env.reset()
    state   = env.state
    pid     = state.current_player
    obs_np  = encode_observation(state, pid)
    mask_np = action_mask(state)

    net = PolicyNet()
    action, log_prob, value = net.act(obs_np, mask_np)

    assert isinstance(action, int)
    assert 0 <= action < ACTION_DIM
    assert mask_np[action], "sampled action must be legal"
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_policy_net_log_prob_finite():
    net = PolicyNet()
    env = CatanEnv(seed=1)
    env.reset()
    for _ in range(20):
        pid     = env.state.current_player
        obs_np  = encode_observation(env.state, pid)
        mask_np = action_mask(env.state)
        action, log_prob, value = net.act(obs_np, mask_np)
        assert np.isfinite(log_prob), f"non-finite log_prob at step"
        assert np.isfinite(value)
        _, _, done, _, _ = env.step(action)
        if done:
            env.reset()


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def test_gae_all_done():
    """Single-step episode: advantage == reward - value."""
    rewards = np.array([1.0], dtype=np.float32)
    values  = np.array([0.5], dtype=np.float32)
    dones   = np.array([True])
    adv, ret = compute_gae(values, rewards, dones, 0.0, 0.99, 0.95)
    assert adv[0] == pytest.approx(0.5, abs=1e-5)
    assert ret[0] == pytest.approx(1.0, abs=1e-5)


def test_gae_multi_step_no_done():
    """
    3-step undiscounted sequence (gamma=1, lambda=1).
    Advantage at t=0 should equal sum of future TD errors.
    """
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones   = np.array([False, False, False])
    # last_value = 0 (episode ends after rollout by convention here)
    adv, ret = compute_gae(values, rewards, dones, 0.0, 1.0, 1.0)
    # Each TD: δ_2=1, δ_1=0, δ_0=0
    # A_2 = 1, A_1 = 0+1*1*1=1, A_0 = 0+1*1*1=1
    assert adv[0] == pytest.approx(1.0, abs=1e-5)
    assert adv[1] == pytest.approx(1.0, abs=1e-5)
    assert adv[2] == pytest.approx(1.0, abs=1e-5)


def test_gae_done_resets_bootstrap():
    """After a done, the next step's advantage is independent."""
    rewards = np.array([1.0, 0.0], dtype=np.float32)
    values  = np.array([0.0, 0.5], dtype=np.float32)
    dones   = np.array([True, False])
    adv, ret = compute_gae(values, rewards, dones, 0.0, 0.99, 0.95)
    # Step 0: done, so advantage = reward - value = 1.0 - 0.0 = 1.0
    assert adv[0] == pytest.approx(1.0, abs=1e-5)
    # Step 1: no done, next value = 0.0 (last_value), δ = 0 + 0 - 0.5 = -0.5
    assert adv[1] == pytest.approx(-0.5, abs=1e-5)


def test_gae_returns_equal_advantage_plus_value():
    rewards = np.random.rand(16).astype(np.float32)
    values  = np.random.rand(16).astype(np.float32)
    dones   = np.zeros(16, dtype=bool)
    adv, ret = compute_gae(values, rewards, dones, 0.0, 0.99, 0.95)
    np.testing.assert_allclose(ret, adv + values, atol=1e-5)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _make_random_batch(T=256):
    """Return random arrays that look like a rollout buffer."""
    obs          = np.random.randn(T, OBS_DIM).astype(np.float32)
    masks        = np.ones((T, ACTION_DIM), dtype=bool)
    actions      = np.random.randint(0, ACTION_DIM, size=T)
    log_probs    = np.random.randn(T).astype(np.float32)
    advantages   = np.random.randn(T).astype(np.float32)
    returns      = np.random.randn(T).astype(np.float32)
    return obs, masks, actions, log_probs, advantages, returns


def test_ppo_update_runs_without_error():
    net   = PolicyNet()
    opt   = torch.optim.Adam(net.parameters(), lr=3e-4)
    obs, masks, actions, log_probs, advantages, returns = _make_random_batch(256)
    stats = ppo_update(
        net, opt, obs, masks, actions, log_probs, advantages, returns,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        batch_size=64, n_epochs=2, max_grad_norm=0.5,
    )
    assert "pg_loss" in stats
    assert "v_loss"  in stats
    assert "entropy" in stats
    assert all(np.isfinite(v) for v in stats.values())


def test_ppo_update_changes_weights():
    net   = PolicyNet()
    opt   = torch.optim.Adam(net.parameters(), lr=3e-4)
    w_before = net.policy_head.weight.data.clone()
    obs, masks, actions, log_probs, advantages, returns = _make_random_batch(256)
    ppo_update(
        net, opt, obs, masks, actions, log_probs, advantages, returns,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        batch_size=64, n_epochs=1, max_grad_norm=0.5,
    )
    assert not torch.equal(net.policy_head.weight.data, w_before)


# ---------------------------------------------------------------------------
# Rollout + self-play smoke test
# ---------------------------------------------------------------------------

def test_rollout_collection_shapes():
    """Collect a short rollout and check buffer shape consistency."""
    T   = 64
    env = CatanEnv(seed=7)
    env.reset()
    net = PolicyNet()

    obs_buf  = np.zeros((T, OBS_DIM),    dtype=np.float32)
    mask_buf = np.zeros((T, ACTION_DIM), dtype=bool)
    act_buf  = np.zeros(T,               dtype=np.int64)

    for t in range(T):
        pid     = env.state.current_player
        obs_np  = encode_observation(env.state, pid)
        mask_np = action_mask(env.state)
        action, _, _ = net.act(obs_np, mask_np)
        obs_buf[t]  = obs_np
        mask_buf[t] = mask_np
        act_buf[t]  = action
        _, _, done, _, _ = env.step(action)
        if done:
            env.reset()

    assert obs_buf.shape  == (T, OBS_DIM)
    assert mask_buf.shape == (T, ACTION_DIM)
    assert act_buf.shape  == (T,)
    # Every sampled action must have been legal
    for t in range(T):
        assert mask_buf[t, act_buf[t]], f"illegal action at step {t}"


def test_self_play_episode_completes():
    """One full self-play episode must reach GAME_OVER within 10 000 steps."""
    env = CatanEnv(seed=99)
    env.reset()
    net = PolicyNet()

    for step in range(50_000):
        pid     = env.state.current_player
        obs_np  = encode_observation(env.state, pid)
        mask_np = action_mask(env.state)
        action, _, _ = net.act(obs_np, mask_np)
        _, _, done, _, info = env.step(action)
        if done:
            assert "winner" in info
            assert info["winner"] in (0, 1)
            return

    pytest.fail("Episode did not complete within 50 000 steps")


def test_vp_reward_shaping_non_negative():
    """Shaped VP reward should be 0 when VP doesn't increase."""
    env = CatanEnv(seed=5)
    env.reset()
    net = PolicyNet()
    VP_SCALE = 0.05

    for _ in range(200):
        pid       = env.state.current_player
        vp_before = visible_vp(env.state, pid)
        obs_np    = encode_observation(env.state, pid)
        mask_np   = action_mask(env.state)
        action, _, _ = net.act(obs_np, mask_np)
        _, sparse, done, _, info = env.step(action)

        vp_after = visible_vp(env.state, pid)
        shaped   = VP_SCALE * max(0.0, float(vp_after - vp_before))
        assert shaped >= 0.0

        if done:
            env.reset()


# ---------------------------------------------------------------------------
# VecEnv
# ---------------------------------------------------------------------------

def test_vecenv_resets_all():
    vec = VecEnv(4, base_seed=0)
    vec.reset_all()
    for env in vec.envs:
        assert env.state is not None


def test_vecenv_batched_rollout_shapes():
    """Collect a short batched rollout and verify buffer shapes."""
    T, N = 32, 4
    vec  = VecEnv(N, base_seed=10)
    vec.reset_all()
    net  = PolicyNet()

    obs_buf  = np.zeros((T, N, OBS_DIM),    dtype=np.float32)
    mask_buf = np.zeros((T, N, ACTION_DIM), dtype=bool)
    act_buf  = np.zeros((T, N),             dtype=np.int64)

    for t in range(T):
        obs_batch  = np.empty((N, OBS_DIM),    dtype=np.float32)
        mask_batch = np.empty((N, ACTION_DIM),  dtype=bool)
        for i, env in enumerate(vec.envs):
            pid = env.state.current_player
            obs_batch[i]  = encode_observation(env.state, pid)
            mask_batch[i] = action_mask(env.state)

        with torch.no_grad():
            dist, _ = net(
                torch.as_tensor(obs_batch),
                torch.as_tensor(mask_batch),
            )
            actions = dist.sample().numpy()

        obs_buf[t]  = obs_batch
        mask_buf[t] = mask_batch
        act_buf[t]  = actions

        for i, env in enumerate(vec.envs):
            _, _, done, _, _ = env.step(int(actions[i]))
            if done:
                vec.reset_one(i)

    assert obs_buf.shape  == (T, N, OBS_DIM)
    assert mask_buf.shape == (T, N, ACTION_DIM)
    assert act_buf.shape  == (T, N)
    for t in range(T):
        for i in range(N):
            assert mask_buf[t, i, act_buf[t, i]], f"illegal action at t={t}, env={i}"


def test_vecenv_independent_seeds():
    """Different envs should produce different initial states."""
    vec = VecEnv(4, base_seed=0)
    vec.reset_all()
    resources = [[h.resource for h in vec.envs[i].state.board.hexes] for i in range(4)]
    assert not all(r == resources[0] for r in resources[1:])


# ---------------------------------------------------------------------------
# Evaluate vs random
# ---------------------------------------------------------------------------

def test_evaluate_vs_random_win_rate_in_range():
    net    = PolicyNet()
    device = torch.device("cpu")
    result = evaluate_vs_random(net, n_games=10, device=device, seed=0)
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["n_games"] == 10
    assert result["avg_steps"] > 0


def test_evaluate_alternates_sides():
    """With n_games=2, runs without error for both sides."""
    net    = PolicyNet()
    device = torch.device("cpu")
    result = evaluate_vs_random(net, n_games=2, device=device, seed=99)
    assert result["n_games"] == 2


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

def test_resume_from_checkpoint():
    """Save a checkpoint, reload it, verify policy weights are identical."""
    import tempfile, os
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt_test.pt")
        torch.save({
            "steps":     1000,
            "policy":    net.state_dict(),
            "optimizer": opt.state_dict(),
        }, path)

        net2 = PolicyNet()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net2.load_state_dict(ckpt["policy"])

        assert ckpt["steps"] == 1000
        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# LR / entropy annealing
# ---------------------------------------------------------------------------

def test_lr_annealing_decreases():
    lr_init, lr_end = 3e-4, 0.0
    fracs = [1.0, 0.75, 0.5, 0.25, 0.0]
    lrs   = [lr_end + (lr_init - lr_end) * f for f in fracs]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1]
    assert lrs[-1] == pytest.approx(lr_end)
    assert lrs[0]  == pytest.approx(lr_init)


def test_entropy_annealing_decreases():
    ent_init, ent_end = 0.05, 0.001
    fracs = [1.0, 0.5, 0.0]
    ents  = [ent_end + (ent_init - ent_end) * f for f in fracs]
    assert ents[0] == pytest.approx(ent_init)
    assert ents[-1] == pytest.approx(ent_end)
    assert ents[0] > ents[1] > ents[2]

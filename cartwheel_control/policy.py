"""The tracking policy: what it observes, what it is asked to track, and the network.

Every 20 ms (50 Hz) the policy sees its last ``hist`` proprioceptive observations and the next
``goal_frames`` frames of the reference motion, and outputs one action per joint: an offset
from the default pose, times the joint's action scale, as the PD position target."""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from cartwheel_control import quat
from cartwheel_control.motion import Motions

GRAVITY = (0.0, 0.0, -1.0)


class RunningNorm(nn.Module):
  """Fixed per-dimension normalization, (x - mean) / std, with the training statistics."""

  def __init__(self, dim: int, eps: float = 1e-2):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(0.0))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / torch.sqrt(self.var + self.eps**2)


def mlp(inp: int, hidden: tuple[int, ...], out: int) -> nn.Sequential:
  layers, d = [], inp
  for h in hidden:
    layers += [nn.Linear(d, h), nn.ELU()]
    d = h
  return nn.Sequential(*layers, nn.Linear(d, out))


class Policy(nn.Module):
  """A motion encoder (goal -> 32-d latent) and a decoder (latent + observation history ->
  action)."""

  def __init__(self, goal_dim: int, obs_dim: int, num_joints: int, locked: tuple[int, ...] = (), latent: int = 32):
    super().__init__()
    self.encoder = mlp(goal_dim, (2048, 1024, 512, 256), latent)
    self.decoder = mlp(latent + obs_dim, (2048, 1024, 512, 256), num_joints)
    self.goal_norm = RunningNorm(goal_dim)
    self.obs_norm = RunningNorm(obs_dim)
    free = torch.ones(num_joints)
    free[list(locked)] = 0
    self.register_buffer("free", free, persistent=False)

  def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(self.encoder(self.goal_norm(goal)))
    return self.decoder(torch.cat([z, self.obs_norm(obs_hist.flatten(-2))], -1)) * self.free


def load(folder: str, device: str, locked_joints: tuple[str, ...] = ()) -> tuple[Policy, dict]:
  """Policy and config.json from a robot folder of the model repo, with the actions of
  locked_joints held at 0."""
  from safetensors.torch import load_file

  with open(os.path.join(folder, "config.json")) as f:
    cfg = json.load(f)
  state = load_file(os.path.join(folder, "policy.safetensors"), device=device)
  goal_dim, obs_dim = state["goal_norm.mean"].shape[0], state["obs_norm.mean"].shape[0]
  locked = tuple(cfg["joints"].index(j) for j in locked_joints)
  pi = Policy(goal_dim, obs_dim, len(cfg["joints"]), locked).to(device)
  pi.load_state_dict(state)
  return pi.eval(), cfg


class Inputs:
  """Observation and goal of a robot state ``s`` = qpos (7 + nj) ++ qvel (6 + nj), as in
  MuJoCo with the root linear velocity in the world frame and the angular velocity in the
  root frame."""

  def __init__(self, motions: Motions, default_joint_pos, goal_frames: int):
    self.motions, self.K = motions, goal_frames
    dev = motions.device
    self.q0 = torch.tensor(default_joint_pos, device=dev)
    self.nj = len(default_joint_pos)
    self.gravity = torch.tensor(GRAVITY, device=dev)

  def proprio(self, s: torch.Tensor, a_prev: torch.Tensor, ref_idx: torch.Tensor) -> torch.Tensor:
    """Joint positions (from the default pose) and velocities, root angular velocity,
    gravity in the root frame, the previous action, and the reference root orientation
    relative to the robot's (6D)."""
    nq = 7 + self.nj
    q_root, w = s[..., 3:7], s[..., nq + 3 : nq + 6]
    grav = quat.apply_inv(q_root, self.gravity.expand_as(w))
    ref_root = self.motions.state(ref_idx)[0][..., 3:7]
    rel = quat.to_6d(quat.mul(quat.conj(q_root), ref_root))
    return torch.cat([s[..., 7:nq] - self.q0, s[..., nq + 6 :], w, grav, a_prev, rel], -1)

  def goal(self, ref_idx: torch.Tensor) -> torch.Tensor:
    """The next K reference frames, from the reference alone: root displacement and
    orientation in the current reference heading frame, root-frame root velocities, joint
    positions and (scaled) velocities."""
    ks = torch.arange(1, self.K + 1, device=ref_idx.device)
    rq, rv = self.motions.state(self.motions.offset(ref_idx.unsqueeze(-1), ks))
    cq = self.motions.state(ref_idx)[0].unsqueeze(-2)
    heading = quat.yaw(cq[..., 3:7]).expand_as(rq[..., 3:7])
    root = rq[..., 3:7]
    feat = torch.cat(
      [
        quat.apply_inv(heading, rq[..., :3] - cq[..., :3]),
        quat.to_6d(quat.mul(quat.conj(heading), root)),
        quat.apply_inv(root, rv[..., :3]),
        rv[..., 3:6],
        rq[..., 7:],
        0.1 * rv[..., 6:],
      ],
      -1,
    )
    return feat.flatten(-2)

  def fell(self, s: torch.Tensor, ref_idx: torch.Tensor, height: float = 0.25, tilt: float = 0.8) -> torch.Tensor:
    """Root height off the reference by more than ``height`` m, or the root's up axis off by
    more than ``tilt`` in gravity z."""
    rq = self.motions.state(ref_idx)[0]
    g = self.gravity.expand_as(rq[..., :3])
    e_tilt = (quat.apply_inv(rq[..., 3:7], g)[..., 2] - quat.apply_inv(s[..., 3:7], g)[..., 2]).abs()
    return ((rq[..., 2] - s[..., 2]).abs() > height) | (e_tilt > tilt)

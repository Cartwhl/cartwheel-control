"""Run a released policy on reference motions in simulation and write a web viewer of the result.

  uv run cartwheel-play --robot rp1
  uv run cartwheel-play --robot asimov-1 --clips dance
  python -m http.server -d play_out/viewer 8000   # then open http://localhost:8000

The policy, the robot model and example motions download from the Hugging Face repo. Each motion
gets its own simulated robot, started in the motion's first pose; all run in parallel on the GPU.
A robot whose root height or tilt leaves the reference's (0.25 m, 0.8 in gravity z) has fallen;
its rollout continues 1 s past the fall so the fall is visible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from cartwheel_control import motion, policy, viewer
from cartwheel_control.robots import ROBOTS

HF_REPO = "andrew-cartwheel/cartwheel-control"


@dataclass
class Args:
  robot: str = "rp1"  # rp1 or asimov-1
  motions: str | None = None  # folder of .npz motions (motion.py); default: the examples in the repo
  clips: str = "."  # regex over motion names
  max_clips: int = 32
  weights: str = HF_REPO  # Hugging Face repo id, or a local copy of it
  out: str = "play_out"
  max_seconds: float = 20.0
  viewer: bool = True
  device: str = "cuda"


def fetch(weights: str, robot: str) -> str:
  """The robot's folder of the model repo, downloading it unless weights is a local copy."""
  if not os.path.isdir(weights):
    from huggingface_hub import snapshot_download

    weights = snapshot_download(weights, allow_patterns=[f"{robot}/*"])
  folder = os.path.join(weights, robot)
  assert os.path.isfile(os.path.join(folder, "config.json")), f"no {robot}/config.json in {weights}"
  return folder


@torch.no_grad()
def rollout(pi, inputs: policy.Inputs, sim, motions: motion.Motions, hist: int, max_seconds: float) -> dict[str, np.ndarray]:
  """One rollout per motion, all in parallel, from each motion's first frame. Returns robot and
  reference qpos per control step and per motion frames / failed / fail_step."""
  n, nq = motions.num_clips, 7 + sim.nj
  dev = motions.device
  idx = motions.clip_start.clone()
  q, v = motions.state(idx)
  sim.reset(q, v)
  steps = int(min(int(motions.clip_len.max()) - 1, max_seconds * motion.FPS))
  end = torch.minimum(motions.clip_len - 1, torch.tensor(steps, device=dev))
  failed = torch.zeros(n, dtype=torch.bool, device=dev)
  fail_step = torch.full((n,), -1, dtype=torch.long, device=dev)
  s = sim.state()
  a = torch.zeros(n, sim.nj, device=dev)
  obs_hist = inputs.proprio(s, a, idx)[:, None].expand(-1, hist, -1).clone()
  robot, ref = [s[:, :nq].cpu()], [q.cpu()]
  for t in range(steps):
    a = pi(obs_hist, inputs.goal(idx))
    s = sim.step(a)
    idx = motions.offset(idx, 1)
    new = (inputs.fell(s, idx) | ~torch.isfinite(s).all(-1)) & ~failed & (t < end)
    fail_step[new] = t + 1
    failed |= new
    robot.append(s[:, :nq].cpu())
    ref.append(motions.state(idx)[0].cpu())
    obs_hist = torch.cat([inputs.proprio(s, a, idx)[:, None], obs_hist[:, :-1]], 1)
  frames = torch.where(failed, torch.minimum(fail_step + int(motion.FPS), end), end) + 1
  return {
    "names": np.array(motions.names),
    "robot": torch.stack(robot, 1).numpy(),
    "ref": torch.stack(ref, 1).numpy(),
    "frames": frames.cpu().numpy(),
    "failed": failed.cpu().numpy(),
    "fail_step": fail_step.cpu().numpy(),
  }


def main():
  a = tyro.cli(Args)
  assert a.robot in ROBOTS, f"--robot is one of {', '.join(ROBOTS)}"
  robot = ROBOTS[a.robot]
  folder = fetch(a.weights, robot.name)
  from cartwheel_control.sim import Sim  # imports mjlab / warp

  files = motion.find(a.motions or os.path.join(folder, "motions"), a.clips, a.max_clips)
  assert files, f"no motion matches {a.clips!r}"
  pi, cfg = policy.load(folder, a.device, robot.locked_joints)
  motions = motion.Motions(files, len(cfg["joints"]), a.device)
  sim = Sim(robot, folder, motions.num_clips, a.device)
  assert sim.joint_names == cfg["joints"], "the policy's joint order differs from the robot model's"
  inputs = policy.Inputs(motions, robot.default_joint_pos, cfg["goal_frames"])
  R = rollout(pi, inputs, sim, motions, cfg["hist"], a.max_seconds)
  os.makedirs(a.out, exist_ok=True)
  npz = os.path.join(a.out, "rollouts.npz")
  np.savez_compressed(npz, **R)
  for name, f, k in zip(R["names"], R["failed"], R["fail_step"]):
    print(f"{'fell at %5.1f s' % (k / motion.FPS) if f else 'completed      '}  {name}")
  print(f"{int(R['failed'].sum())} of {len(files)} fell; rollouts in {npz}")
  if a.viewer:
    out = os.path.join(a.out, "viewer")
    viewer.write(out, robot, folder, R)
    print(f"viewer: python -m http.server -d {out} 8000, then open http://localhost:8000")


if __name__ == "__main__":
  main()

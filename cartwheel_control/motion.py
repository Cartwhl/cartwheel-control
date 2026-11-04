"""Reference motions: robot joint trajectories at 50 Hz, one .npz per motion with ``qpos``
(frames, 7 + joints): root position, root orientation (wxyz) and joint angles in the MJCF's
joint order. Frames of all motions are concatenated and addressed by a global frame index."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

from cartwheel_control import quat

FPS = 50.0
DT = 1.0 / FPS


def find(folder: str, pattern: str = ".", limit: int | None = None) -> list[str]:
  """Sorted .npz files in folder whose name matches the regex."""
  files = sorted(glob.glob(os.path.join(folder, "*.npz")))
  files = [f for f in files if re.search(pattern, os.path.basename(f)[:-4])]
  return files[:limit]


class Motions:
  def __init__(self, files: list[str], num_joints: int, device: str | torch.device):
    clips = []
    for f in files:
      z = np.load(f)
      assert float(z.get("fps", FPS)) == FPS, f"{f}: motions must be at {FPS:g} Hz"
      q = np.asarray(z["qpos"], np.float32)
      assert q.ndim == 2 and q.shape[1] == 7 + num_joints, f"{f}: qpos is {q.shape}, expected (frames, {7 + num_joints})"
      clips.append(q)
    lengths = np.array([len(q) for q in clips])
    self.names = [os.path.basename(f)[:-4] for f in files]
    self.device = torch.device(device)
    self.qpos = torch.from_numpy(np.concatenate(clips)).to(self.device)
    self.qpos[:, 3:7] = quat.normalize(self.qpos[:, 3:7])
    self.clip_start = torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)[:-1]])).long().to(self.device)
    self.clip_len = torch.from_numpy(lengths).long().to(self.device)
    self.num_clips = len(clips)
    self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)

  def clip_end(self, idx: torch.Tensor) -> torch.Tensor:
    """Last global frame index of the motion containing idx."""
    clip = self.frame_clip[idx]
    return self.clip_start[clip] + self.clip_len[clip] - 1

  def offset(self, idx: torch.Tensor, k: torch.Tensor | int) -> torch.Tensor:
    """idx + k, held at the motion's last frame."""
    return torch.minimum(idx + k, self.clip_end(idx))

  def state(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """qpos (..., 7 + joints) and qvel (..., 6 + joints) at global frame idx, velocities by
    central differences; root angular velocity in the root frame (MuJoCo's free joint)."""
    clip = self.frame_clip[idx]
    lo = self.clip_start[clip]
    hi = lo + self.clip_len[clip] - 1
    ip, inx = (idx - 1).clamp(min=lo), (idx + 1).clamp(max=hi)
    span = ((inx - ip).float() * DT).unsqueeze(-1)
    q, qp, qn = self.qpos[idx], self.qpos[ip], self.qpos[inx]
    lin = (qn[..., :3] - qp[..., :3]) / span
    ang = quat.to_rotvec(quat.mul(quat.conj(qp[..., 3:7]), qn[..., 3:7])) / span
    jv = (qn[..., 7:] - qp[..., 7:]) / span
    return q, torch.cat([lin, ang, jv], -1)

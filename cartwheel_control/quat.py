"""Quaternion helpers (wxyz, batched over leading dimensions)."""

from __future__ import annotations

import torch


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    -1,
  )


def conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], -1)


def apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
  return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return apply(conj(q), v)


def to_rotvec(q: torch.Tensor) -> torch.Tensor:
  q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
  xyz = q[..., 1:]
  s = torch.linalg.norm(xyz, dim=-1, keepdim=True)
  angle = 2.0 * torch.atan2(s, q[..., :1])
  small = s < 1e-6
  k = torch.where(small, 2.0 + angle**2 / 12.0, angle / torch.where(small, 1.0, s))
  return xyz * k


def to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ],
    -1,
  ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def to_6d(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix."""
  return to_mat(q)[..., :, :2].reshape(*q.shape[:-1], 6)


def yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  a = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  zero = torch.zeros_like(a)
  return torch.stack([torch.cos(a / 2), zero, zero, torch.sin(a / 2)], -1)


def normalize(q: torch.Tensor) -> torch.Tensor:
  return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

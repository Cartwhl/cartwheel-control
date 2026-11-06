"""Web viewer of rollouts: the robot from robot.glb, posed per frame, with the reference motion as
a translucent ghost. A folder of index.html, robot.glb, manifest.json and poses.bin
(per motion: frames x (reference bodies + robot bodies) x int16 [position mm, quaternion * 32767],
MuJoCo body order); serve it over HTTP."""

from __future__ import annotations

import json
import os
import shutil
import struct

import mujoco
import numpy as np

from cartwheel_control.robots import Robot

HERE = os.path.dirname(os.path.abspath(__file__))


def body_poses(m: mujoco.MjModel, d: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
  """qpos -> (bodies, 7) int16 [pos mm, quat * 32767] of every body but the world."""
  d.qpos[:] = qpos
  mujoco.mj_kinematics(m, d)
  out = np.empty((m.nbody - 1, 7), np.int16)
  out[:, :3] = np.clip(np.round(d.xpos[1:] * 1000), -32767, 32767)
  out[:, 3:] = np.round(d.xquat[1:] * 32767)
  return out


def swatch(glb: str) -> list[str]:
  """sRGB hex of the first two materials' base colors, for the legend."""
  with open(glb, "rb") as f:
    f.seek(12)
    n, _ = struct.unpack("<II", f.read(8))
    mats = json.loads(f.read(n))["materials"]
  srgb = lambda c: round(255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055))  # noqa: E731
  return ["#%02x%02x%02x" % tuple(srgb(c) for c in mt["pbrMetallicRoughness"]["baseColorFactor"][:3]) for mt in mats[:2]]


def write(out: str, robot: Robot, folder: str, R: dict[str, np.ndarray], fps: float = 25.0) -> None:
  m = robot.model(folder)
  d = mujoco.MjData(m)
  stride = int(round(50 / fps))
  clips, blob = [], bytearray()
  for c, name in enumerate(R["names"]):
    n, failed = int(R["frames"][c]), bool(R["failed"][c])
    live = int(R["fail_step"][c]) if failed else n  # root error over the frames before a fall
    root_err = np.linalg.norm(R["robot"][c, :live, :3] - R["ref"][c, :live, :3], axis=-1)
    offset = len(blob)
    sel = np.arange(0, n, stride)
    for t in sel:
      blob.extend(body_poses(m, d, R["ref"][c, t]).tobytes())
      blob.extend(body_poses(m, d, R["robot"][c, t]).tobytes())
    clips.append(
      {
        "name": str(name),
        "offset": offset,
        "frames": len(sel),
        "seconds": round(n / 50, 2),
        "fell_at": round(int(R["fail_step"][c]) / 50, 2) if failed else None,
        "mpjpe_mm": round(float(R["mpjpe_mm"][c]), 1),
        "root_err_cm": [round(100 * float(root_err.mean()), 1), round(100 * float(root_err.max()), 1)],
      }
    )
  os.makedirs(out, exist_ok=True)
  glb = os.path.join(folder, "robot.glb")
  shutil.copy(glb, os.path.join(out, "robot.glb"))
  manifest = {
    "title": f"{robot.title} · Cartwheel Control",
    "robot": robot.title,
    "swatch": swatch(glb),
    "fps": 50 / stride,
    "bodies": [m.body(b).name for b in range(1, m.nbody)],
    "clips": clips,
  }
  with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f)
  with open(os.path.join(out, "poses.bin"), "wb") as f:
    f.write(blob)
  shutil.copy(os.path.join(HERE, "viewer.html"), os.path.join(out, "index.html"))
  print(f"poses {len(blob) / 1e6:.1f} MB, {len(clips)} motions -> {out}")

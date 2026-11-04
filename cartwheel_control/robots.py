"""The robots a policy is released for: their MJCF (downloaded with the policy), PD gains, action
scales and default pose, as the policy was trained with them."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import mujoco

DAMPING_RATIO = 2.0


@dataclass(frozen=True)
class Joint:
  kp: float
  kd: float
  effort: float
  armature: float
  scale: float  # action -> position target offset (rad per unit)


@dataclass(frozen=True)
class Robot:
  name: str  # folder in the Hugging Face repo
  title: str
  mjcf: str  # inside the robot folder
  default_joint_pos: tuple[float, ...]  # joint order; the action offset and observation reference
  standing_height: float  # root height in the default pose, soles on the floor
  foot_geoms: str  # regex
  joints: Callable[[mujoco.MjModel], dict[str, Joint]]
  locked_joints: tuple[str, ...] = ()  # the policy's action there is always 0 (the PD holds the default pose)

  def model(self, folder: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(os.path.join(folder, self.mjcf))


def armature_and_effort(m: mujoco.MjModel) -> dict[str, tuple[float, float]]:
  """joint name -> (armature, effort limit): from the motor ctrlrange, or for a model without
  motors from the joint's actuatorfrcrange."""
  out = {}
  if m.nu:
    for a in range(m.nu):
      j = m.actuator_trnid[a, 0]
      out[m.joint(j).name] = (float(m.dof_armature[m.jnt_dofadr[j]]), float(m.actuator_ctrlrange[a, 1]))
  else:
    for j in range(1, m.njnt):
      out[m.joint(j).name] = (float(m.dof_armature[m.jnt_dofadr[j]]), float(m.jnt_actfrcrange[j, 1]))
  return out


# ---------------------------------------------------------------- RoboParty RP1
# kp = armature * w^2 and kd = 2 * zeta * armature * w, at 10 Hz.
RP1_W = 10 * 2.0 * 3.1415926535


def _rp1_joints(m: mujoco.MjModel) -> dict[str, Joint]:
  w = RP1_W
  return {
    n: Joint(kp=arm * w**2, kd=2.0 * DAMPING_RATIO * arm * w, effort=eff, armature=arm, scale=0.25 * eff / (arm * w**2))
    for n, (arm, eff) in armature_and_effort(m).items()
  }


RP1 = Robot(
  name="rp1",
  title="RoboParty RP1",
  mjcf="model/mjcf/robot.xml",
  default_joint_pos=(
    -0.312, 0, 0, -0.669, -0.363, 0,  0.312, 0, 0, 0.669, -0.363, 0,  0, 0,
    0.2, 0.2, 0, -0.6, 0, 0, 0,  -0.2, -0.2, 0, 0.6, 0, 0, 0,
  ),  # fmt: skip
  standing_height=0.754,
  foot_geoms=r"^(left|right)_foot_.*_collision$",
  joints=_rp1_joints,
)


# ---------------------------------------------------------------- Menlo Asimov-1
# Each joint's action scale is set by its kind; kp = 0.25 * effort / scale and
# kd = 2 * zeta * sqrt(kp * armature).
ASIMOV_SCALE = {
  "hip_pitch": 0.548, "hip_yaw": 0.548, "waist_yaw": 0.548,
  "hip_roll": 0.351, "knee": 0.351,
  "ankle_pitch": 0.439, "ankle_roll": 0.439,
  "shoulder": 0.439, "elbow": 0.439, "wrist_yaw": 0.439, "neck": 0.439,
}  # fmt: skip


def _asimov_joints(m: mujoco.MjModel) -> dict[str, Joint]:
  out = {}
  for n, (arm, eff) in armature_and_effort(m).items():
    scale = next(s for k, s in ASIMOV_SCALE.items() if k in n)
    kp = 0.25 * eff / scale
    out[n] = Joint(kp=kp, kd=2.0 * DAMPING_RATIO * math.sqrt(kp * arm), effort=eff, armature=arm, scale=scale)
  return out


ASIMOV = Robot(
  name="asimov-1",
  title="Menlo Asimov-1",
  mjcf="model/xml/asimov_1_25dof.xml",
  default_joint_pos=(
    -0.166, -0.019, -0.006, 0.332, -0.168, -0.019,  0.166, 0.019, 0.006, -0.332, 0.168, 0.019,  0, 0, 0,
    -0.207, 0.193, 0.058, -0.779, -0.157,  0.207, -0.193, -0.058, 0.779, 0.157,
  ),  # fmt: skip
  standing_height=0.623,
  foot_geoms=r"^(left|right)_foot[1-4]_collision$",
  joints=_asimov_joints,
  # the reference motions hold the head straight and nothing tracks it, so the neck is held
  locked_joints=("neck_yaw_joint", "neck_pitch_joint"),
)

# ---------------------------------------------------------------- Unitree G1 (stock rubber hands)
# mjlab's G1 actuators follow the RP1 rule (10 Hz, damping ratio 2) with the armature and effort
# limit per joint, which g1.xml carries, so the RP1 gain function applies unchanged.
G1 = Robot(
  name="g1",
  title="Unitree G1",
  mjcf="model/g1.xml",
  default_joint_pos=(
    -0.312, 0, 0, 0.669, -0.363, 0,  -0.312, 0, 0, 0.669, -0.363, 0,  0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,  0.2, -0.2, 0, 0.6, 0, 0, 0,
  ),  # fmt: skip
  standing_height=0.76,
  # the rubber palms and fingers make contact like the feet, so they get the same friction
  foot_geoms=r"^(left|right)_(foot[1-7]|palm|fingers)_collision$",
  joints=_rp1_joints,
)

ROBOTS = {r.name: r for r in (RP1, ASIMOV, G1)}

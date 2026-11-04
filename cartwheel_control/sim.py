"""Batched physics in MuJoCo Warp through mjlab: one simulated robot per motion on flat ground,
200 Hz physics, PD position actuators, the policy acting at 50 Hz. No randomization: foot
friction 1.0, no noise."""

from __future__ import annotations

import os
import re

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from cartwheel_control import quat
from cartwheel_control.robots import Robot

FRICTION = 1.0


def _spec(xml: str) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(xml)
  # mjlab adds its own position actuators and ground: drop the MJCF's motors, sensors and floor
  for a in list(spec.actuators):
    spec.delete(a)
  for s in list(spec.sensors):
    spec.delete(s)
  for g in list(spec.worldbody.geoms):
    spec.delete(g)
  return spec


def _entity(robot: Robot, folder: str, joint_names: list[str]) -> tuple[EntityCfg, dict[str, float]]:
  """mjlab robot config and per-joint action scales."""
  xml = os.path.join(folder, robot.mjcf)
  joints = robot.joints(robot.model(folder))
  feet = robot.foot_geoms
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0, 0, robot.standing_height),
      joint_pos=dict(zip(joint_names, robot.default_joint_pos)),
      joint_vel={".*": 0.0},
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=0,  # no self-collision
        conaffinity=1,
        condim={feet: 3, ".*_collision": 1},
        priority={feet: 1, ".*": 0},
        friction={feet: (FRICTION,)},
      ),
    ),
    spec_fn=lambda: _spec(xml),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(
        BuiltinPositionActuatorCfg(
          target_names_expr=(re.escape(n),), stiffness=j.kp, damping=j.kd, effort_limit=j.effort, armature=j.armature
        )
        for n, j in joints.items()
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  ), {re.escape(n): j.scale for n, j in joints.items()}


class Sim:
  def __init__(self, robot: Robot, folder: str, num_envs: int, device: str):
    m = robot.model(folder)
    self.joint_names = [m.joint(j).name for j in range(1, m.njnt)]
    self.nj = len(self.joint_names)
    lo, hi = torch.tensor(m.jnt_range[1:], dtype=torch.float32, device=device).unbind(-1)
    mid, half = 0.5 * (lo + hi), 0.45 * (hi - lo)
    self.soft_lo, self.soft_hi = mid - half, mid + half
    entity, scale = _entity(robot, folder, self.joint_names)
    cfg = ManagerBasedRlEnvCfg(
      scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=num_envs, entities={"robot": entity}),
      observations={
        "actor": ObservationGroupCfg(terms={"joint_pos": ObservationTermCfg(func=mdp_obs.joint_pos_rel)}, concatenate_terms=True)
      },
      actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=scale, use_default_offset=True)},
      commands={},
      events={},
      rewards={},
      terminations={"time_out": TerminationTermCfg(func=mdp_term.time_out, time_out=True)},
      sim=SimulationCfg(nconmax=128, njmax=500, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
      decimation=4,
      episode_length_s=1e9,
    )
    self.env = ManagerBasedRlEnv(cfg, device=device)
    self.robot = self.env.scene["robot"]
    names = [n.split("/")[-1] for n in self.robot.joint_names]
    assert names == self.joint_names, f"mjlab joint order {names} differs from the MJCF's"
    self.n, self.device = num_envs, torch.device(device)

  def state(self) -> torch.Tensor:
    """(envs, qpos ++ qvel), positions relative to each env's origin."""
    d = self.env.sim.data
    qpos, qvel = d.qpos[:, : 7 + self.nj], d.qvel[:, : 6 + self.nj]
    return torch.cat([qpos[:, :3] - self.env.scene.env_origins, qpos[:, 3:], qvel], -1)

  def reset(self, qpos: torch.Tensor, qvel: torch.Tensor):
    """Every robot to the given state (root angular velocity in the root frame)."""
    env, ids = self.env, torch.arange(self.n, device=self.device)
    env.extras["log"] = {}
    env._reset_idx(ids)
    pos = qpos[:, :3] + env.scene.env_origins
    ang_w = quat.apply(qpos[:, 3:7], qvel[:, 3:6])
    jp = torch.clamp(qpos[:, 7:], self.soft_lo, self.soft_hi)
    self.robot.write_joint_state_to_sim(jp, qvel[:, 6:], env_ids=ids)
    self.robot.write_root_state_to_sim(torch.cat([pos, qpos[:, 3:7], qvel[:, :3], ang_w], -1), env_ids=ids)
    self.robot.reset(env_ids=ids)
    env.scene.write_data_to_sim()

  @torch.no_grad()
  def step(self, action: torch.Tensor) -> torch.Tensor:
    """One 20 ms control step (4 physics steps); returns the new state."""
    env = self.env
    env.action_manager.process_action(action)
    for _ in range(env.cfg.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
    return self.state()

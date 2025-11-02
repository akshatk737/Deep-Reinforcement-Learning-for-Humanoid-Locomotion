import  gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time

class HumanoidWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, render=False, time_step=1./240., urdf_folder="urdfs"):
        super().__init__()
        self.render = render
        self.time_step = time_step
        self.urdf_folder = urdf_folder

        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")
        self.urdf_files = [os.path.join(self.urdf_folder, f)
                           for f in os.listdir(self.urdf_folder)
                           if f.endswith(".urdf")]

        if not self.urdf_files:
            raise FileNotFoundError(f"No URDFs found in {self.urdf_folder}")

        self.humanoids = []
        for path in self.urdf_files:
            self.humanoids.append(p.loadURDF(path, [0, 0, 1]))

        self.humanoid = self.humanoids[0]
        self.num_joints = p.getNumJoints(self.humanoid)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.num_joints,), dtype=np.float32)
        obs_dim = self.num_joints * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        joint_states = p.getJointStates(self.humanoid, range(self.num_joints))
        joint_positions = np.array([s[0] for s in joint_states])
        joint_velocities = np.array([s[1] for s in joint_states])
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        torso_height = np.array([base_pos[2]])
        return np.concatenate([joint_positions, joint_velocities, torso_height])

    def reset(self, *, seed=None, options=None, initial_pose=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        if self.urdf_files:
            self.humanoid = p.loadURDF(self.urdf_files[0], [0, 0, 1])

        self.num_joints = p.getNumJoints(self.humanoid)

        if initial_pose is not None and len(initial_pose) == self.num_joints:
            for i, angle in enumerate(initial_pose):
                p.resetJointState(self.humanoid, i, angle)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        max_torque = 50.0
        torques = max_torque * action

        for i, torque in enumerate(torques):
            p.setJointMotorControl2(bodyIndex=self.humanoid,
                                    jointIndex=i,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=torque)

        p.stepSimulation()

        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        base_vel, _ = p.getBaseVelocity(self.humanoid)
        forward_vel = base_vel[0]

        alive_bonus = 0.05
        energy_penalty = 0.001 * np.sum(np.square(action))
        reward = forward_vel + alive_bonus - energy_penalty

        done = base_pos[2] < 0.5
        obs = self._get_obs()
        info = {"forward_vel": forward_vel}
        return obs, reward, done, False, info

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

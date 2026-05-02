import mujoco
import mujoco.viewer
import numpy as np
from utils import ik_step

class FrankaEnv:
    def __init__(self, render_mode=None):
        # 1. Core Model Loading
        self.model = mujoco.MjModel.from_xml_path("./franka_emika_panda/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        self.arm_dof = 7
        self.gripper_site_id = self.model.site('hand_tcp').id
        self.cube1_body_id = self.model.body("cube1").id
        self.target_body_id = self.model.body("target").id
        self.initial_angles = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])
        self.starting_target_dist = 0.0


        # IK control parameters
        self.target_pos = None
        self.target_quat = None
        
        # 2. RL Specific Settings
        self.render_mode = render_mode
        self.viewer = None
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[:self.arm_dof] = self.initial_angles
        self.data.ctrl[:self.arm_dof] = self.initial_angles
        
        
        # Table center coordinates
        table_x, table_y, table_z = 0.4, 0.3, 0.32 

        for cube_name in ["cube1", "cube2", "cube3"]:
            joint_id = self.model.joint(f"{cube_name}_joint").qposadr[0]
            
            # 1. Randomize X and Y around the table center
            new_x = table_x + np.random.uniform(-0.1, 0.1)
            new_y = table_y + np.random.uniform(-0.1, 0.1)
            
            # 2. Apply to qpos. 
            # A freejoint has 7 qpos values: [x, y, z, qw, qx, qy, qz]
            self.data.qpos[joint_id] = new_x
            self.data.qpos[joint_id + 1] = new_y
            self.data.qpos[joint_id + 2] = table_z # Keep it on the table surface
            
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize IK targets
        self.target_pos = self.data.site_xpos[self.gripper_site_id].copy()
        self.target_quat = np.array([0, 1.0, 0, 0])
        
        self.starting_target_dist = np.linalg.norm(self.data.xpos[self.cube1_body_id] - self.data.xpos[self.target_body_id])

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos[:self.arm_dof].copy() # 7
        qvel = self.data.qvel[:self.arm_dof].copy() # 7
        gripper_open = np.array([1.0 if self.data.ctrl[7] > 127 else 0.0]) # 1
        
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy() # 3
        target_pos = self.data.xpos[self.target_body_id].copy() # 3
        
        cube1_pos = self.data.body("cube1").xpos.copy() # 3
        
        # Relative vectors
        rel_gripper_to_cube = cube1_pos - gripper_pos # 3
        rel_cube_to_target = target_pos - cube1_pos # 3

        # Total size: 7 + 7 + 1 + 3 + 3 + 3 + 3 + 3 = 30
        return np.concatenate([
            qpos, qvel, gripper_open, 
            gripper_pos, target_pos, cube1_pos,
            rel_gripper_to_cube, rel_cube_to_target
        ]).astype(np.float32)

    def step(self, action):
        """
        Applies an action and advances the simulation.
        action: np.array of size (5,) -> [dx, dy, dz, d_yaw, gripper_state]
        """
        # 1. Apply Actions
        dx, dy, dz, d_yaw, gripper = action
        
        # Update target position
        self.target_pos += np.array([dx, dy, dz])
        
        # Reset target if it drifts too far from current position e.g., due to collisions or IK failures
        if self._collision_check():
            self.target_pos = self.data.site_xpos[self.gripper_site_id].copy()
        
        # Update target orientation
        if d_yaw != 0.0:
            angle = d_yaw
            axis_of_rotation = np.array([0.0, 0.0, 1.0])
            delta_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta_quat, axis_of_rotation, angle)
            
            new_target_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_target_quat, delta_quat, self.target_quat)
            self.target_quat = new_target_quat / np.linalg.norm(new_target_quat)
        
        # Compute inverse kinematics
        dq = ik_step(self.model, self.data, self.gripper_site_id, self.target_pos, self.target_quat)
        self.data.ctrl[:7] = self.data.qpos[:7] + dq
        self.data.ctrl[7] = 255.0 if gripper > 0.5 else 0.0
        
        # 2. Step Physics
        num_substeps = 20
        for _ in range(num_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # 3. Get results
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_done()
        
        return obs, reward, terminated, False, {}

    def _calculate_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        cube_pos = self.data.xpos[self.cube1_body_id]
        target_pos = self.data.xpos[self.target_body_id]

        reach_dist = np.linalg.norm(gripper_pos - cube_pos)
        target_dist = np.linalg.norm(cube_pos - target_pos)

        reward = 0.0
        if self._is_grabbing():
            reward += 2.0  # Reward the act of holding the object
            
            # 2. Lift 
            if not self._is_cube_touching_table():
                reward += cube_pos[2] * 100.0  # Table height is 0.32m
            
            # 3. Carrying Reward: Reward getting closer to the target while holding the cube
            reward += (self.starting_target_dist - target_dist) * 300.0
            
        else:
            # Reaching reward if not grabbing
            reward -= reach_dist

        if self._collision_check():
            reward -= 50.0  # Penalize collisions harshly

        # 4. SUCCESS TERMINAL REWARD ---
        target_proximity_reward = 1.0 / (1.0 + target_dist**2)
        reward += target_proximity_reward
        
        # Adjust threshold slightly to account for the Z-height difference 
        # of the cube resting on top of the target box.
        if target_dist < 0.05:
            reward += 1000.0 
        
        return reward

    def _check_done(self):
        cube_pos = self.data.xpos[self.cube1_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        
        target_dist = np.linalg.norm(cube_pos - target_pos)

        return target_dist < 0.05

    def _is_grabbing(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        cube_pos = self.data.xpos[self.cube1_body_id]
        
        if np.linalg.norm(gripper_pos - cube_pos) < 0.02:
            if self.data.ctrl[7] < 0.01 and self.data.qpos[7] > 0.01:
                #print(f"Gripper is grabbing, Gripper dist to cube: {np.linalg.norm(gripper_pos - cube_pos):.4f}")
                return True
            
        return False
    
    def _collision_check(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        tracking_error = np.linalg.norm(self.target_pos - gripper_pos)

        if tracking_error > 0.04: 
            #print(f"Arm stuck! Error: {tracking_error:.3f}")
            return True
            
        return False
    
    def _is_cube_touching_table(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if not geom1_name or not geom2_name:
                continue
                
            # Check if this specific contact pair is cube1 and the table
            is_cube = "cube1" in geom1_name or "cube1" in geom2_name
            is_table = "table" in geom1_name or "table" in geom2_name
            
            if is_cube and is_table:
                return True
        
        epsilon = 0.001
        if 0.32 - epsilon < self.data.xpos[self.cube1_body_id][2] < 0.32 + epsilon:
            #if the cube is at table height and without vertical velocity, we can assume it's resting on the table even if contacts are missed
            if abs(self.data.cvel[self.cube1_body_id][5]) < 0.01:
                return True

        #print(f"Cube has lifted off the table!, cube height: {self.data.xpos[self.cube1_body_id][2]:.3f}")
        return False

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
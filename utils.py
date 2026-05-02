import numpy as np
import mujoco
import pygame

def compute_pose_error(current_pos, current_quat, target_pos, target_quat):
    # Position error
    pos_error = target_pos - current_pos

    # Quaternion error
    neg_quat = np.zeros(4)
    mujoco.mju_negQuat(neg_quat, current_quat)

    error_quat = np.zeros(4)
    mujoco.mju_mulQuat(error_quat, target_quat, neg_quat)

    rot_error = np.zeros(3)
    mujoco.mju_quat2Vel(rot_error, error_quat, 1.0)

    return np.concatenate([pos_error, rot_error])

def compute_jacobian(model, data, site_id, dofs):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return np.vstack([jacp[:, :dofs], jacr[:, :dofs]])

def damped_least_squares(J, error, damping=0.05):
    A = J @ J.T + (damping ** 2) * np.eye(6)
    f = np.linalg.solve(A, error)
    dq = J.T @ f
    return dq

def ik_step(model, data, site_id, target_pos, target_quat, dofs=7, damping=0.05):
    # Current pose
    current_pos = data.site_xpos[site_id].copy()
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])

    # Error
    error = compute_pose_error(current_pos, current_quat, target_pos, target_quat)

    # Jacobian
    J = compute_jacobian(model, data, site_id, dofs)

    # Solve IK
    dq = damped_least_squares(J, error, damping)

    return dq

def get_keyboard_commands(viewer, gripper_open):
    velocity_vector = np.zeros(3)
    angular_velocity = 0.0 

    # Handle closing the window and gripper toggle
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            viewer.close()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                gripper_open = not gripper_open
            if event.key == pygame.K_r:
                return velocity_vector, angular_velocity, gripper_open, True 
                
    # Get keyboard state
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:    velocity_vector[0] = 1.0  # +X
    if keys[pygame.K_s]:    velocity_vector[0] = -1.0 # -X
    if keys[pygame.K_a]:    velocity_vector[1] = 1.0  # +Y
    if keys[pygame.K_d]:    velocity_vector[1] = -1.0 # -Y
    if keys[pygame.K_UP]:   velocity_vector[2] = 1.0  # +Z
    if keys[pygame.K_DOWN]: velocity_vector[2] = -1.0 # -Z
    if keys[pygame.K_LEFT]:  angular_velocity = 1.0   # Rotate +Z
    if keys[pygame.K_RIGHT]: angular_velocity = -1.0  # Rotate -Z

    return velocity_vector, angular_velocity, gripper_open, False
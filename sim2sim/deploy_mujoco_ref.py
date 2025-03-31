import os
utils_path = os.path.abspath(__file__ + "../utils/")
import sys
sys.path.append(utils_path)
import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
from collections import deque

from omegaconf import OmegaConf

from motion_lib.motion_lib_robot import MotionLibRobot

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

if __name__ == "__main__":
    # get config file name from command line
    with open("configs/g1_ref.yaml") as file:
        config = OmegaConf.load(file)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        ref_joint_angles_scale = config["ref_joint_angles_scale"]
        ref_joint_velocities_scale = config["ref_joint_velocities_scale"]
        action_scale = config["action_scale"]

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_history = config["frame_stack"]
        
        # motion
        motion_lib = MotionLibRobot(config["motion"], num_envs=1, device="cpu")
        motion_lib.load_motions(random_sample=False)
        motion_len = motion_lib.get_motion_length([0])
        motion_start_times = torch.zeros(1, dtype=torch.float32, device="cpu")
        motion_dt = motion_lib._motion_dt
        motion_start_idx = 0
        num_motions = motion_lib._num_unique_motions
        motion_res = motion_lib.get_motion_state([0], torch.tensor([0.]))

        joint_limit_lo = config["dof_pos_lower_limit_list"]
        joint_limit_hi = config["dof_pos_upper_limit_list"]
        soft_dof_pos_limit = 0.98
        for i in range(len(joint_limit_lo)):
            # soft limits
            if i != 5 and i != 11 and i !=4 and i != 10:
                m = (joint_limit_lo[i] + joint_limit_hi[i]) / 2
                r = joint_limit_hi[i] - joint_limit_lo[i]
                joint_limit_lo[i] = m - 0.5 * r * soft_dof_pos_limit
                joint_limit_hi[i] = m + 0.5 * r * soft_dof_pos_limit
        torque_limits = config["dof_effort_limit_list"]

    while True:

        # Load robot model
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        m.opt.timestep = simulation_dt
        reset = True
        # load policy
        policy = torch.jit.load(policy_path)

        with mujoco.viewer.launch_passive(m, d) as viewer:
            for _ in range(28):
                add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
            viewer.user_scn.geoms[27].pos = [0,0,0]
            while viewer.is_running():
                if reset:
                    # define context variables
                    action = np.zeros(num_actions, dtype=np.float32)
                    target_dof_pos = default_angles.copy()
                    obs = np.zeros(num_obs*num_history, dtype=np.float32)
                    hist_obs = np.zeros(num_obs*(num_history-1), dtype=np.float32)
                    hist_dict = {
                        "actions": np.zeros(num_actions*(num_history-1), dtype=np.float32),
                        "base_ang_vel": np.zeros(3*(num_history-1), dtype=np.float32),
                        "dof_pos": np.zeros(num_actions*(num_history-1), dtype=np.float32),
                        "dof_vel": np.zeros(num_actions*(num_history-1), dtype=np.float32),
                        "projected_gravity": np.zeros(3*(num_history-1), dtype=np.float32),
                        "ref_joint_angles": np.zeros(num_actions*(num_history-1), dtype=np.float32),
                        "ref_joint_velocities": np.zeros(num_actions*(num_history-1), dtype=np.float32),
                        "ref_motion_phase": np.zeros(1*(num_history-1), dtype=np.float32),
                    }
                    counter = 0
                    d.qpos[:3] = motion_res['root_pos'][0]
                    d.qpos[3:7] = torch.cat([motion_res['root_rot'][0][3:], motion_res['root_rot'][0][:3]])
                    d.qpos[7:] = motion_res['dof_pos'][0]
                    d.qvel[:3] = motion_res['root_vel'][0]
                    d.qvel[3:6] = motion_res['root_ang_vel'][0]
                    d.qvel[6:] = motion_res['dof_vel'][0]
                    
                    # d.qpos[:3] = [0, 0, 0.78]
                    # d.qpos[3:7] = [0.996, 0, 0.087, 0.]
                    # d.qpos[3:7] = [1., 0, 0., 0.]
                    # d.qpos[7:7+15] = default_angles[:15]
                    d.qvel[:] = 0
                    reset = False
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                counter += 1

                motion_res_cur = motion_lib.get_motion_state([0], (counter + control_decimation) * simulation_dt + motion_start_times)

                ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
                ref_joint_pos = motion_res_cur["dof_pos"][0]
                ref_joint_vel = motion_res_cur["dof_vel"][0]


                for i in range(ref_body_pos_extend.shape[0]):
                # if i in [0, 1, 4, 7, 2, 5, 8, 16, 18, 22, 17, 19, 23, 15]:  # joint for matching
                    viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i] + torch.tensor([1., 0., 0.])
                if counter % control_decimation == 0:
                    
                    # Apply control signal here.

                    # create observation
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    # reference motion
                    ref_joint_angles = ref_joint_pos - d.qpos[7:]
                    ref_joint_velocities = ref_joint_vel - d.qvel[6:]
                    count = counter * simulation_dt
                    motion_times = (counter + control_decimation) * simulation_dt + motion_start_times
                    ref_motion_phase = motion_times / motion_len

                    obs[:num_actions] = action
                    obs[num_actions:num_actions+3] = omega
                    obs[num_actions+3 : num_actions*2+3] = qj
                    obs[num_actions*2+3 : num_actions*3+3] = dqj
                    history_numpy = []
                    for key in sorted(hist_dict.keys()):
                        history_numpy.append(hist_dict[key])
                    obs[num_actions*3+3 : num_actions*3+3+num_obs*(num_history-1)] = np.concatenate(history_numpy, axis=-1)
                    obs[num_actions*3+3+num_obs*(num_history-1): num_actions*3+6+num_obs*(num_history-1)] = gravity_orientation
                    obs[num_actions*3+6+num_obs*(num_history-1): num_actions*4+6+num_obs*(num_history-1)] = ref_joint_angles
                    obs[num_actions*4+6+num_obs*(num_history-1): num_actions*5+6+num_obs*(num_history-1)] = ref_joint_velocities
                    obs[num_actions*5+6+num_obs*(num_history-1):] = ref_motion_phase


                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    
                    hist_dict["actions"] = np.concatenate([action, hist_dict["actions"][:-num_actions]])
                    hist_dict["base_ang_vel"] = np.concatenate([omega, hist_dict["base_ang_vel"][:-3]])
                    hist_dict["dof_pos"] = np.concatenate([qj, hist_dict["dof_pos"][:-num_actions]])
                    hist_dict["dof_vel"] = np.concatenate([dqj, hist_dict["dof_vel"][:-num_actions]])
                    hist_dict["projected_gravity"] = np.concatenate([gravity_orientation, hist_dict["projected_gravity"][:-3]])
                    hist_dict["ref_joint_angles"] = np.concatenate([ref_joint_angles, hist_dict["ref_joint_angles"][:-num_actions]])
                    hist_dict["ref_joint_velocities"] = np.concatenate([ref_joint_velocities, hist_dict["ref_joint_velocities"][:-num_actions]])
                    hist_dict["ref_motion_phase"] = np.concatenate([ref_motion_phase, hist_dict["ref_motion_phase"][:-1]])
                    
                    # policy inference
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    # transform action to target_dof_pos
                    actions_scaled = action * action_scale
                    # target_dof_pos = np.clip(target_dof_pos, joint_limit_lo, joint_limit_hi)
                    p_limits_low = (-np.array(torque_limits)) + kps*dqj
                    p_limits_high = (np.array(torque_limits)) + kds*dqj
                    actions_low = (p_limits_low/kps) - default_angles + d.qpos[7:]
                    actions_high = (p_limits_high/kps) - default_angles + d.qpos[7:]
                    target_dof_pos = np.clip(actions_scaled, actions_low, actions_high) + default_angles
                    # target_dof_pos = actions_scaled + default_angles

        
                    if ((np.array(d.qpos[7:])-np.array(joint_limit_lo))<0).sum() >0 or ((np.array(d.qpos[7:])-np.array(joint_limit_hi))>0).sum() > 0:
                        print("Joint limit reached")
                        if ((np.array(d.qpos[7:])-np.array(joint_limit_lo))<0).sum() >0:
                            idx = np.where((np.array(d.qpos[7:])-np.array(joint_limit_lo))<0)[0]
                            print("Low limit Joint index: ", idx, d.qpos[7:][idx], np.array(joint_limit_lo)[idx])
                        if ((np.array(d.qpos[7:])-np.array(joint_limit_hi))>0).sum() > 0:
                            idx = np.where((np.array(d.qpos[7:])-np.array(joint_limit_hi))>0)[0]
                            print("High limit Joint index: ", idx, d.qpos[7:][idx], np.array(joint_limit_hi)[idx])
                        reset = False

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()


                current_time = counter * simulation_dt + motion_start_times
                if current_time > motion_len:
                    reset = True
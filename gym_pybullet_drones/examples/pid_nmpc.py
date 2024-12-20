"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along circular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

# Import pyQuaternion for handling quaternions
from pyquaternion import Quaternion as pyQuaternion

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 2
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), 
    #                        R*np.sin((i/6)*2*np.pi+np.pi/2)-R, 
    #                        H+i*H_STEP] for i in range(num_drones)])
    INIT_XYZS = np.array([[0, 0, 0] for i in range(num_drones)])
    print(f'INIT_XYZS: \n{INIT_XYZS}')
    print(f'INIT_XYZS shape: \n{INIT_XYZS.shape}')
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    INIT_Q = np.array([pyQuaternion(axis=[0, 0, 1], angle=INIT_RPYS[j, 2]) * 
                    pyQuaternion(axis=[0, 1, 0], angle=INIT_RPYS[j, 1]) * 
                    pyQuaternion(axis=[1, 0, 0], angle=INIT_RPYS[j, 0]) 
                    for j in range(num_drones)])

    # Define colors for drones
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]

    #### Generate Waypoints with Positions and Orientations ######
    PERIOD = 1  # Duration over which waypoints are defined
    NUM_WP = control_freq_hz * PERIOD  # Number of waypoints

    # Define final positions and orientations for each drone
    FINAL_P = np.array([[0.0, 0.0, 1.0] for _ in range(num_drones)])  # Example final positions
    FINAL_Q = np.array([pyQuaternion(axis=[0, 0, 1], angle=0) for _ in range(num_drones)])  # Example final orientations

    # Initialize lists to store waypoints for each drone
    positions = [np.linspace(INIT_XYZS[j], FINAL_P[j], NUM_WP) for j in range(num_drones)]
    orientations = [np.array([pyQuaternion.slerp(INIT_Q[j],
                                               FINAL_Q[j],
                                               t).elements 
                              for t in np.linspace(0, 1, NUM_WP)]) for j in range(num_drones)]
    
    # Initialize waypoint counters for each drone
    wp_counters = np.array([0 for _ in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        # initial_xyzs=INIT_XYZS,
                        # initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    # Optionally, visualize waypoints at the start
    # print(f'positions: \n{positions}')
    # waypoint_radius = 2e-3  # Adjust size as needed
    # for j in range(num_drones):
    #     for wp in positions[j]:
    #         waypoint_visual = p.createVisualShape(
    #             shapeType=p.GEOM_SPHERE,
    #             radius=waypoint_radius,
    #             rgbaColor=colors[j % len(colors)] + [1]
    #         )
    #         _ = p.createMultiBody(
    #             baseMass=0,
    #             baseVisualShapeIndex=waypoint_visual,
    #             basePosition=wp
    #         )

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        # #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ > 5 and i%10 == 0 and i/env.SIM_FREQ < 10:
        #     p.loadURDF("duck_vhacd.urdf", 
        #               [0 + random.gauss(0, 0.3),
        #                -0.5 + random.gauss(0, 0.3),
        #                3],
        #               p.getQuaternionFromEuler([random.uniform(0, 2*np.pi),
        #                                        random.uniform(0, 2*np.pi),
        #                                        random.uniform(0, 2*np.pi)]),
        #               physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            if wp_counters[j] < NUM_WP:
                # Extract target position from waypoints
                target_pos = positions[j][wp_counters[j]]
                
                # Extract target orientation from waypoints and convert to RPY
                target_quat = orientations[j][wp_counters[j]]
                target_rpy = pyQuaternion(target_quat).yaw_pitch_roll  # Returns (yaw, pitch, roll)
                target_rpy = np.array(target_rpy[::-1])  # Convert to (roll, pitch, yaw)
            else:
                # If all waypoints are exhausted, maintain the final waypoint
                target_pos = FINAL_P[j]
                target_rpy = np.array([0, 0, 0])  # Adjust as needed

            # Compute control action using PID controller
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                                        control_timestep=env.CTRL_TIMESTEP,
                                        state=obs[j],
                                        target_pos=target_pos,
                                        target_rpy=target_rpy
                                    )

        #### Go to the next way point #####################
        for j in range(num_drones):
            if wp_counters[j] < NUM_WP -1:
                wp_counters[j] += 1  # Move to the next waypoint

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([
                           positions[j][wp_counters[j]],
                           pyQuaternion(orientations[j][wp_counters[j]]).yaw_pitch_roll,
                           np.zeros(6)  # Placeholder for additional control variables if needed
                       ])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

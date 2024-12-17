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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.pytypes import DroneParameters

from gym_pybullet_drones.control.NMPCControl import NMPCPlanner, nmpcConfig

from pyquaternion import Quaternion as pyQuaternion

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120 #48
DEFAULT_DURATION_SEC = 2.0#0.5
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
    INIT_XYZS = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
    INIT_RPYS = np.array([0.0, 0.0, 0.0]).reshape(1, 3)

    # Define colors for drones
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]

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

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    #### Get the drone parameters ###############################
    drone_params = DroneParameters()
    drone_params.initialize_from_env(env=env)
    print(f'max rpm: {drone_params.max_rpm}')
    print(f'max thrust: {drone_params.max_thrust}')

    #### Initialize the trajectory #############################
    NUM_WAYPOINTS = 25

    initial_state = env._getDroneStateVector(0)
    INIT_P = initial_state[0:3]
    INIT_Q = pyQuaternion(initial_state[3:7])
    INIT_V = initial_state[10:13]
    INIT_W = initial_state[13:16]
    # INIT_T = drone_params.thrust_coefficient * initial_state[16:]**2
    INIT_T = np.zeros(4)

    FINAL_P = np.array([0.0, 0.0, 1.0])
    FINAL_Q = pyQuaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    FINAL_V = np.zeros(3)
    FINAL_W = np.zeros(3)
    FINAL_T = np.full(4, drone_params.mass*env.G/4)

    t_values = np.linspace(0, 1, NUM_WAYPOINTS)
    
    positions = np.linspace(INIT_P, FINAL_P, NUM_WAYPOINTS)
    orientations = np.array([pyQuaternion.slerp(INIT_Q, FINAL_Q, t).elements for t in t_values])
    velocities = np.linspace(INIT_V, FINAL_V, NUM_WAYPOINTS)
    angular_velocities = np.linspace(INIT_W, FINAL_W, NUM_WAYPOINTS)
    thrusts = np.linspace(INIT_T, FINAL_T, NUM_WAYPOINTS)

    # print(f'positions: \n{positions}')
    # print(f'orientations: \n{orientations}')
    # print(f'velocities: \n{velocities}')
    # print(f'angular_velocities: \n{angular_velocities}')
    # print(f'thrusts: \n{thrusts}')

    waypoints = np.hstack([positions, orientations, velocities, angular_velocities, thrusts])
    # print(f'waypoints: \n{waypoints}')
    # print(f'waypoints shape: \n{waypoints.shape}')

    # Optionally, visualize waypoints at the start
    print(f'positions shape: \n{positions.shape}')
    waypoint_radius = 2e-3  # Adjust size as needed
    for j in range(num_drones):
        for wp in positions:
            waypoint_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=waypoint_radius,
                rgbaColor=colors[j % len(colors)] + [1]
            )
            _ = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=waypoint_visual,
                basePosition=wp
            )

    #### Initialize the controller #############################

    # Custom definitions for all matrices
    # Qpk = np.diag([800.0, 800.0, 800.0])*1e+4 # Higher weights on position error
    # Qxyk = np.array([60.0])*1e+4  # Orientation cost (xy-plane)
    # Qzk = np.array([60.0])*1e+2 # Orientation cost (z-axis)
    # Qvk = np.diag([1.0, 1.0, 1.0])*1e+1  # Velocity cost
    # Qwk = np.diag([0.5, 0.5, 0.1])*1e+1  # Angular velocity cost
    # Qtk = np.diag([3.0, 3.0, 3.0, 3.0])*1e+3  # Thrust cost
    # Quk = np.diag([1.0, 1.0, 1.0, 1.0])*1e+3 # Control input cost
    # Rk = np.diag([1.0, 1.0, 1.0, 1.0])*1e+3  # Input cost

    Qpk = np.diag([800.0, 800.0, 800.0])*1e+6 # Higher weights on position error
    Qxyk = np.array([60.0])*1e+4  # Orientation cost (xy-plane)
    Qzk = np.array([60.0])*1e+2 # Orientation cost (z-axis)
    Qvk = np.diag([1.0, 1.0, 1.0])*1e+2  # Velocity cost
    Qwk = np.diag([0.5, 0.5, 0.1])*1e+2  # Angular velocity cost
    Qtk = np.diag([3.0, 3.0, 3.0, 3.0])*1e+5  # Thrust cost
    Quk = np.diag([1.0, 1.0, 1.0, 1.0])*1e+5 # Control input cost
    Rk = np.diag([1.0, 1.0, 1.0, 1.0])*1e+3  # Input cost

    # Create an nmpcConfig instance with custom matrices
    config = nmpcConfig(
        TK=40,#40,
        Qpk=Qpk,
        Qxyk=Qxyk,
        Qzk=Qzk,
        Qvk=Qvk,
        Qwk=Qwk,
        Qtk=Qtk,
        Quk=Quk,
        Rk=Rk,
        DTK=1e-3#0.05  # Custom time step
    )
    nmpc_planner = NMPCPlanner(config=config,
                               drone_params=drone_params,
                               env=env,
                               waypoints=waypoints)
    nmpc_planner.mpc_prob_init()

    #### Initialize variables ###################################
    action = np.zeros((num_drones, 4))
    START = time.time()

    #### Run the simulation ####################################
    trigger_time = int(0.15 * env.CTRL_FREQ)
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        # #### Compute control using NMPC ###########################
        for j in range(num_drones):

            current_state = obs[j]
            print(f'current state: {current_state}')
            optimal_u = abs(nmpc_planner.plan(current_state))
            rpms = np.sqrt(optimal_u / drone_params.thrust_coefficient)
            print(f'rpms before clipping: {rpms}')

            if not np.isnan(rpms).any():
                action[j, :] = rpms.reshape(1, 4)

            # if (i == trigger_time):
            #     drone_params.G[:, 1] *= 0.0
            #     drone_params.en_rot[1] *= 0.0
            #     config.Qzk *= 0.

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=obs[j],
                    #    control=action[j, :]
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
    logger.save_as_csv("nmpc") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Flight test for fault tolerant control using nmpc')
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
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
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
DEFAULT_DURATION_SEC = 1.0
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

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
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
    p_ref = np.array([1.0, 1.0, 0.0])  # desired position
    q_ref = np.array([1.0, 0.0, 0.0, 0.0]) # desired orientation (no rotation)
    v_ref = np.zeros(3) 
    w_ref = np.zeros(3)
    t_ref = np.full(4, drone_params.mass*9.81/4)  # hover thrust: each rotor lifts 1/4 m*g

    waypoints = np.hstack([p_ref, q_ref, v_ref, w_ref, t_ref])

    #### Initialize the controller #############################

    # Custom definitions for all matrices
    Qpk = np.diag([80.0, 80.0, 800.0])  # Higher weights on position error
    Qxyk = np.array([60.0])  # Orientation cost (xy-plane)
    Qzk = np.array([60.0])*0  # Orientation cost (z-axis)
    Qvk = np.diag([1.0, 1.0, 1.0])  # Velocity cost
    Qwk = np.diag([0.5, 0.5, 0.1])  # Angular velocity cost
    Qtk = np.diag([3.0, 3.0, 3.0, 3.0])  # Thrust cost
    Quk = np.diag([1.0, 1.0, 1.0, 1.0]) # Control input cost
    Rk = np.diag([1.0, 1.0, 1.0, 1.0])*1000  # Input cost

    # Create an nmpcConfig instance with custom matrices
    config = nmpcConfig(
        TK=40,
        Qpk=Qpk,
        Qxyk=Qxyk,
        Qzk=Qzk,
        Qvk=Qvk,
        Qwk=Qwk,
        Qtk=Qtk,
        Quk=Quk,
        Rk=Rk,
        DTK=0.05  # Custom time step
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
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        # #### Compute control using NMPC ###########################
        for j in range(num_drones):
            current_state = obs[j]
            print(f'current state: {current_state}')
            optimal_u = nmpc_planner.plan(current_state) + 1e-8
            print(f'optimal_u: {optimal_u}')
            rpms = np.sqrt(optimal_u / drone_params.thrust_coefficient)
            print(f'rpms before clipping: {rpms}')

            if not np.isnan(rpms).any():
                action[j, :] = rpms.reshape(1, 4)
        
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
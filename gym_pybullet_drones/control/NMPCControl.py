from dataclasses import dataclass, field
import numpy as np
import casadi as ca

from gym_pybullet_drones.utils.rotations import Quaternion
from gym_pybullet_drones.utils.pytypes import DroneParameters

@dataclass
class nmpcConfig:
    NXK: int = 17  # length of state vector: z = [p, q, v, w, t]
    NU: int = 4  # length of input vector: u = = [t1, t2, t3, t4]
    TK: int = 20  # finite time horizon length

    Rk: list = field(
        default_factory=lambda: np.diag([0.5, 0.5, 0.5, 0.5])
    )  # input cost matrix for inputs

    Qpk: list = field(
        default_factory=lambda: np.diag([100.0, 100.0, 100.0])
    ) # state error cost matrix (position)
    Qxyk: list = field(
        default_factory=lambda: np.array([10.0])
    ) # state error cost matrix (Orientation - xy)
    Qzk: list = field(
        default_factory=lambda: np.array([0.0])
    ) # state error cost matrix (Orientation - z)
    Qvk: list = field(
        default_factory=lambda: np.diag([50.0, 50.0, 50.0])
    ) # state error cost matrix (velocity)
    Qwk: list = field(
        default_factory=lambda: np.diag([1.0, 1.0, 1.0])
    ) # state error cost matrix (angular velocity)
    Qtk: list = field(
        default_factory=lambda: np.diag([0.1, 0.1, 0.1, 0.1])
    ) # state error cost matrix (thrust)
    Quk: list = field(
        default_factory=lambda: np.diag([0.01, 0.01, 0.01, 0.01])
    ) # state error cost matrix (control)

    DTK: float = 0.1  # time step [s] kinematic

class NMPCPlanner:
    """
    """
    def __init__(self,
                 config : nmpcConfig,
                 drone_params : DroneParameters,
                 env, 
                 waypoints):
        self.config = config
        self.drone_params = drone_params
        self.env = env
        self.waypoints = waypoints
        self.last_state = None

    def _get_curr_state(self):
        """
        """
        curr_state = self.env._getDroneStateVector(0)
        return curr_state

    def _get_Gt(self, G, t):
        """
        """
        Gt = G @ t
        T = Gt[0]
        tau = Gt[1:]
        return T, tau

    def _get_current_waypoint(self, lookahead_distance, current_state):
        waypoints = np.array(self.waypoints)

        if waypoints.ndim == 1:
            return waypoints
        elif waypoints.ndim == 2:
            for _, wp in enumerate(waypoints):
                distance = np.linalg.norm(wp[:3] - current_state[:3])
                if distance > lookahead_distance:
                    return wp
            return waypoints[-1]
        else:
            raise ValueError("Waypoints array must be either 1D (single waypoint) \
                             or 2D (multiple waypoints).")
        
    def get_cost_vector(self, state, ref_state, control):
        # Extract components of the state
        p = state[0:3]
        q = state[3:7]
        v = state[7:10]
        w = state[10:13]
        t = state[13:]
        u = control

        # Extract components of the reference state
        p_ref = ref_state[0:3]    
        q_ref = ref_state[3:7]    
        v_ref = ref_state[7:10]  
        w_ref = ref_state[10:13]
        t_ref = ref_state[13:17]
        u_ref = ref_state[13:17]

        # Compute the quaternion split components
        q_z, q_xy = Quaternion.solve_quaternion_split(q=q, q_ref=q_ref)

        # Compute the errors
        p_error = p - p_ref
        v_error = v - v_ref
        w_error = w - w_ref
        t_error = t - t_ref
        u_error = u - u_ref

        y = ca.vertcat(p_error, q_xy[1]**2 + q_xy[2]**2, q_z[3], v_error, w_error, t_error, u_error)

        return y

    def mpc_prob_init(self):

        # State variables
        p = ca.SX.sym('p', 3)  # Position (x, y, z) in world frame
        q = ca.SX.sym('q', 4)  # Quaternion
        v = ca.SX.sym('v', 3)  # Velocity in world frame
        w = ca.SX.sym('w', 3)  # Angular velocity in body frame
        t = ca.SX.sym('t', 4)  # Rotor thrusts

        z = ca.vertcat(p, q, v, w, t)
        n_states = z.numel()

        # Control input
        u = ca.SX.sym('u', 4)  # Rotor commands
        n_controls = u.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        Z = ca.SX.sym('Z', n_states, self.config.TK + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', n_controls, self.config.TK)

        # coloumn vector for storing initial state and target state
        P = ca.SX.sym('P', n_states, self.config.TK + 1)

        # cost state weights matrix
        Q = ca.diagcat(
            ca.DM(self.config.Qpk),  
            ca.DM([[self.config.Qxyk[0]]]),
            ca.DM([[self.config.Qzk[0]]]),
            ca.DM(self.config.Qvk),       
            ca.DM(self.config.Qwk),
            ca.DM(self.config.Qtk),
            ca.DM(self.config.Quk)
        )

        # final state weight matrix
        Qn = Q * 10

        # controls weights matrix
        R = ca.diagcat(*np.diag(self.config.Rk))

        # -------------------------------------------
        # Build the dynamics
        # -------------------------------------------

        # Translational dynamics
        quat_q = Quaternion(scalar=q[0], vec=q[1:4])
        R_rot = quat_q.to_rotm(q)
        T = ca.sum1(t)
        p_dot = v
        v_dot = R_rot @ ca.vertcat(0, 0, T) - ca.vertcat(0, 0, 9.81)

        # Rotational dynamics
        # curr_state = self._get_curr_state()
        # wf = ca.SX(curr_state[13:16])
        # tf = self.drone_params.thrust_coefficient * (ca.SX(curr_state[16:])**2)
        # G = ca.SX(self.drone_params.get_control_effectiveness_matrix())
        # _, tauf = self._get_Gt(G, tf)
        # T, tau = self._get_Gt(G, t)

        # if (self.last_state is None):
        #     wf_dot = wf / self.drone_params.dt
        # else:
        #     last_wf = ca.SX(self.last_state[13:16])
        #     wf_dot = (wf - last_wf) / self.drone_params.dt
        # self.last_state = curr_state

        # I = ca.SX(self.drone_params.inertia_matrix)
        # w_dot = ca.inv(I) @ (I@wf_dot + tau - tauf)

        G = ca.SX(self.drone_params.get_control_effectiveness_matrix())
        I = ca.SX(self.drone_params.inertia_matrix)
        _, tau = self._get_Gt(G, t)
        w_dot = ca.inv(I) @ (tau - ca.cross(w, I @ w))

        # Quaternion dynamics   
        quat_w = Quaternion(scalar=ca.SX(0), vec=w)
        q_dot = 0.5 * (quat_q.__mul__(quat_w)).q

        # Actuator dynamics
        t_dot = (u - t) / self.drone_params.dt

        z_dot = ca.vertcat(p_dot, q_dot, v_dot, w_dot, t_dot)

        # maps states, controls to dynamics
        f = ca.Function('f', [z, u], [z_dot])

        # -------------------------------------------
        # Build the NMPC problem
        # -------------------------------------------
        cost_fn = 0  # cost function
        g = Z[:, 0] - P[:, 0]  # z(0) = z0 constraint in the equation

        # runge kutta
        for k in range(self.config.TK):
            state = Z[:, k]
            control = U[:, k]
            ref_state = P[:, k]
            cost_state = self.get_cost_vector(state=state, ref_state=ref_state, control=control)
            cost_fn = cost_fn \
                + (cost_state).T @ Q @ (cost_state) \
                + (control- ref_state[13:]).T @ R @ (control - ref_state[13:])
            
            next_state = Z[:, k+1]
            k1 = f(state, control)
            k2 = f(state + self.config.DTK * (k1/2), control)
            k3 = f(state + self.config.DTK * (k2/2), control)
            k4 = f(state + self.config.DTK * k3, control)
            next_state_RK4 = state + (self.config.DTK / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, next_state - next_state_RK4)

        terminal_state = Z[:, -1]
        terminal_ref_state = P[:, -1]
        terminal_control = U[:, -1]
        terminal_cost_state = self.get_cost_vector(state=terminal_state, 
                                                   ref_state=terminal_ref_state, 
                                                   control=terminal_control)
        cost_fn = cost_fn + (terminal_cost_state).T @ Qn @ (terminal_cost_state)

        OPT_variables = ca.vertcat(
            Z.reshape((-1, 1)),
            U.reshape((-1, 1))
        )

        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        ipopt_opts = {
            'ipopt': {
                'print_level': 3,
                'max_iter': 2000,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
                'warm_start_init_point': 'yes',
            },
            'print_time': 1,
        }

        # Solver initialization, this is the main solver for the NMPC problem which will be called at each time step
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, ipopt_opts)

        N = n_states * (self.config.TK + 1) + n_controls * self.config.TK
        lbx = -ca.inf * ca.DM.ones((N, 1))
        ubx = ca.inf * ca.DM.ones((N, 1))

        lbx[n_states*(self.config.TK+1)+0::n_controls]         = 0
        lbx[n_states*(self.config.TK+1)+1::n_controls]         = 0 
        lbx[n_states*(self.config.TK+1)+2::n_controls]         = 0
        lbx[n_states*(self.config.TK+1)+3::n_controls]         = 0 
        ubx[n_states*(self.config.TK+1)+0::n_controls]         = self.drone_params.max_thrust/4 
        ubx[n_states*(self.config.TK+1)+1::n_controls]         = self.drone_params.max_thrust/4 
        ubx[n_states*(self.config.TK+1)+2::n_controls]         = self.drone_params.max_thrust/4 
        ubx[n_states*(self.config.TK+1)+3::n_controls]         = self.drone_params.max_thrust/4 

        # lbg is all zeros
        lbg = ca.vertcat(
            ca.DM.zeros((n_states*(self.config.TK+1), 1)),
        )
        ubg = ca.vertcat(
            ca.DM.zeros((n_states*(self.config.TK+1), 1)),
        )

        # store the arguments for the solver, these are updated at each time step
        self.args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }
        self.U0 = ca.DM.zeros((n_controls, self.config.TK))

    def mpc_prob_solve(self, goal_state, x0):
        
        curr_state = ca.vertcat(
            x0[0:3],    # position
            x0[3:7],    # quaternion
            x0[10:13],  # velocity
            x0[13:16],  # angular velocity
            self.drone_params.thrust_coefficient * (x0[16:])**2,   # thrust
            )

        self.args['p'] = ca.horzcat(
            curr_state,    # current state
            ca.repmat(goal_state, 1, self.config.TK)   # target state
        )

        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(ca.repmat(curr_state, 1, self.config.TK+1), self.config.NXK*(self.config.TK+1), 1),
            ca.reshape(self.U0, self.config.NU*self.config.TK, 1)
        )

        sol = self.solver(
            x0=self.args['x0'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            p=self.args['p']
        )

        u_sol = ca.reshape(sol['x'][self.config.NXK*(self.config.TK+1):], self.config.NU, self.config.TK)
        x_sol = ca.reshape(sol['x'][:self.config.NXK*(self.config.TK+1)], self.config.NXK, self.config.TK+1)

        self.ot = u_sol[:, 0].full().flatten()

        print(f'u_sol shape: {u_sol.shape}')
        print(f'u_sol[:, 0] shape: {u_sol[:, 0].shape}')

        # Return the first control action
        return self.ot

    def plan(self, current_state):
        """
        """
        v = current_state[10:13]  # velocity
        v_mag = np.linalg.norm(v)
        v_lookahead = max(v_mag, 0.1)  # Avoid zero lookahead

        lookahead_distance = v_lookahead * (self.config.TK * self.config.DTK)
        goal_state = self._get_current_waypoint(lookahead_distance, current_state)
        print(f'goal_state: {goal_state}')
        optimal_u = self.mpc_prob_solve(goal_state, current_state)

        return optimal_u
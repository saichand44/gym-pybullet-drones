from dataclasses import dataclass, field
import numpy as np

from dataclasses import dataclass, field
import numpy as np

@dataclass
class PythonMsg:
    '''
    base class for creating types and messages in python
    '''
    def __setattr__(self,key,value):
        '''
        Overloads default atribute-setting functionality
          to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally
          adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if key not in self.__dataclass_fields__:
            raise TypeError(f'Cannot add new field "{key}" to frozen class {self.__class__.__name__}')
        else:
            object.__setattr__(self, key, value)

@dataclass
class DroneParameters(PythonMsg):
    """
    Dataclass for drone configuration and parameters.
    """
    drone_name: str = field(default="GenericDrone")  # Name of the drone model
    mass: float = field(default=0.027)  # Mass of the drone (kg)
    arm_length: float = field(default=0.0397)  # Distance from the center to the rotor (m)
    thrust_to_weight_ratio: float = field(default=2.25)  # Thrust-to-weight ratio
    inertia_matrix: np.ndarray = field(default_factory=lambda: np.diag([1.4e-5, 1.4e-5, 2.17e-5]))  # Inertia matrix
    thrust_coefficient: float = field(default=3.16e-10)  # Rotor thrust coefficient (N/(rad/s)²)
    torque_coefficient: float = field(default=7.94e-12)  # Rotor torque coefficient (Nm/(rad/s)²)
    collision_height: float = field(default=0.025)  # Collision height (m)
    collision_radius: float = field(default=0.06)  # Collision radius (m)
    collision_z_offset: float = field(default=0.0)  # Z-offset for collision (m)
    max_speed_kmh: float = field(default=30.0)  # Maximum speed (km/h)
    ground_effect_coefficient: float = field(default=11.36859)  # Ground effect coefficient
    propeller_radius: float = field(default=0.0231348)  # Propeller radius (m)
    drag_coefficients: np.ndarray = field(default_factory=lambda: np.array([9.1785e-7, 9.1785e-7, 10.311e-7]))  # Drag coefficients
    downwash_coefficient_1: float = field(default=2267.18)  # Downwash coefficient 1
    downwash_coefficient_2: float = field(default=0.16)  # Downwash coefficient 2
    downwash_coefficient_3: float = field(default=-0.11)  # Downwash coefficient 3
    dt: float = field(default=0.1) # simulation interval
    max_rpm: float = field(default=0.0)  # maximum rpm
    max_thrust: float = field(default=0.0)  # maximum thrust
    G: np.ndarray = field(default_factory=lambda: np.eye(4))  # G
    en_rot: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1]))
    
    def initialize_from_env(self, env):
        """
        Initializes parameters from the given environment instance.
        """
        self.drone_name = env.DRONE_MODEL.value
        self.mass = env.M
        self.arm_length = env.L
        self.thrust_to_weight_ratio = env.THRUST2WEIGHT_RATIO
        self.inertia_matrix = env.J
        self.thrust_coefficient = env.KF
        self.torque_coefficient = env.KM
        self.collision_height = env.COLLISION_H
        self.collision_radius = env.COLLISION_R
        self.collision_z_offset = env.COLLISION_Z_OFFSET
        self.max_speed_kmh = env.MAX_SPEED_KMH
        self.ground_effect_coefficient = env.GND_EFF_COEFF
        self.propeller_radius = env.PROP_RADIUS
        self.drag_coefficients = env.DRAG_COEFF
        self.downwash_coefficient_1 = env.DW_COEFF_1
        self.downwash_coefficient_2 = env.DW_COEFF_2
        self.downwash_coefficient_3 = env.DW_COEFF_3
        self.max_rpm = env.MAX_RPM  # Maximum RPM of the propeller  
        self.max_thrust = env.MAX_THRUST  # Total maximum thrust  
        self.G = self.get_control_effectiveness_matrix()

    def get_inverse_inertia_matrix(self):
        """
        Returns the inverse of the inertia matrix.
        """
        return np.linalg.inv(self.inertia_matrix)

    def get_control_effectiveness_matrix(self):
        """
        Returns the control effectiveness matrix
        """
        if (self.drone_name == "cf2x"):
            angles = np.array([7*np.pi/4, 3*np.pi/4, 5*np.pi/4, np.pi/4])
        elif (self.drone_name == "cf2p"):
            angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        else:
            raise ValueError("Control effectiveness matrix is defined only for 'cf2x' and 'cf2p'")

        r = np.array([[self.arm_length*np.cos(theta), self.arm_length*np.sin(theta)] for theta in angles])
        r_x = r[:, 0]  # x-coordinates
        r_y = r[:, 1]  # y-coordinates
    
        G = np.array([[1, 1, 1, 1],
                        [r_y[0], r_y[1], r_y[2], r_y[3]],
                        [-r_x[0], -r_x[1], -r_x[2], -r_x[3]],
                        [-self.torque_coefficient, 
                         -self.torque_coefficient, 
                         self.torque_coefficient,
                         self.torque_coefficient]])

        # G = np.array([[1, 1, 1, 1],
        #                 [-r_y[0], r_y[1], -r_y[2], r_y[3]],
        #                 [-r_x[0], r_x[1], r_x[2], -r_x[3]],
        #                 [-self.torque_coefficient, 
        #                  -self.torque_coefficient, 
        #                  self.torque_coefficient,
        #                  self.torque_coefficient]])
        
        print(f'G matrix: \n{G}')
        return G

    def get_motor_thrust_limits(self):
        """
        Returns the thrust limits of the motors
        """
        return self.max_thrust*np.ones(4) 
    
    def get_motor_rpm_limits(self):
        """
        Returns the RPM limits of the motors
        """
        return self.max_rpm*np.ones(4)
    
    def print_parameters(self):
        """
        Prints all parameters for verification.
        """
        print(f"Drone Model: {self.drone_name}")
        print("Drone Parameters:")
        print(f"Mass: {self.mass} kg")
        print(f"Arm Length: {self.arm_length} m")
        print(f"Thrust-to-Weight Ratio: {self.thrust_to_weight_ratio}")
        print(f"Inertia Matrix: \n{self.inertia_matrix}")
        print(f"Thrust Coefficient (k_f): {self.thrust_coefficient} N/(rad/s)²")
        print(f"Torque Coefficient (k_m): {self.torque_coefficient} Nm/(rad/s)²")
        print(f"Collision Height: {self.collision_height} m")
        print(f"Collision Radius: {self.collision_radius} m")
        print(f"Collision Z Offset: {self.collision_z_offset} m")
        print(f"Max Speed: {self.max_speed_kmh} km/h")
        print(f"Ground Effect Coefficient: {self.ground_effect_coefficient}")
        print(f"Propeller Radius: {self.propeller_radius} m")
        print(f"Drag Coefficients: {self.drag_coefficients}")
        print(f"Downwash Coefficient 1: {self.downwash_coefficient_1}")
        print(f"Downwash Coefficient 2: {self.downwash_coefficient_2}")
        print(f"Downwash Coefficient 3: {self.downwash_coefficient_3}")
import casadi as ca

class Quaternion:
    def __init__(self, scalar=ca.SX(1), vec=ca.SX([0, 0, 0])): 
        # Ensure inputs are in SX format
        self.q = ca.vertcat(scalar, vec)

    def normalize(self):
        self.q = ca.if_else(self.q[0] < 0, -self.q, self.q)
        norm = ca.sqrt(ca.sumsqr(self.q))
        self.q = self.q / norm

    def scalar(self):
        return self.q[0]

    def vec(self):
        return self.q[1:4]

    def axis_angle(self):
        theta = 2 * ca.acos(self.scalar())
        vec = self.vec()
        norm_vec = ca.norm_2(vec)
        if norm_vec == 0:
            return ca.SX([0, 0, 0])
        vec = vec / norm_vec
        return vec * theta

    def euler_angles(self):
        phi = ca.atan2(2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
                       1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2))
        theta = ca.asin(2 * (self.q[0] * self.q[2] - self.q[3] * self.q[1]))
        psi = ca.atan2(2 * (self.q[0] * self.q[3] + self.q[1] * self.q[2]),
                       1 - 2 * (self.q[2] ** 2 + self.q[3] ** 2))
        return ca.vertcat(phi, theta, psi)

    def from_axis_angle(self, a):
        angle = ca.norm_2(a)
        if angle != 0:
            axis = a / angle
        else:
            axis = ca.SX([1, 0, 0])
        self.q[0] = ca.cos(angle / 2)
        self.q[1:4] = axis * ca.sin(angle / 2)

    def from_rotm(self, R):
        theta = ca.acos((ca.trace(R) - 1) / 2)
        omega_hat = (R - ca.transpose(R)) / (2 * ca.sin(theta))
        omega = ca.SX([omega_hat[2, 1], -omega_hat[2, 0], omega_hat[1, 0]])
        self.q[0] = ca.cos(theta / 2)
        self.q[1:4] = omega * ca.sin(theta / 2)
        self.normalize()

    def to_rotm(self, q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        R = ca.SX(3, 3)
        R[0, 0] = 2 * (qw**2 + qx**2) - 1
        R[0, 1] = 2 * (qx * qy - qw * qz)
        R[0, 2] = 2 * (qx * qz + qw * qy)
        R[1, 0] = 2 * (qx * qy + qw * qz)
        R[1, 1] = 2 * (qw**2 + qy**2) - 1
        R[1, 2] = 2 * (qy * qz - qw * qx)
        R[2, 0] = 2 * (qx * qz - qw * qy)
        R[2, 1] = 2 * (qy * qz + qw * qx)
        R[2, 2] = 2 * (qw**2 + qz**2) - 1
        return R

    def inv(self):
        q_inv = Quaternion(self.scalar(), -self.vec())
        q_inv.normalize()
        return q_inv

    def __mul__(self, other):
        t0 = self.q[0] * other.q[0] - \
             self.q[1] * other.q[1] - \
             self.q[2] * other.q[2] - \
             self.q[3] * other.q[3]
        t1 = self.q[0] * other.q[1] + \
             self.q[1] * other.q[0] + \
             self.q[2] * other.q[3] - \
             self.q[3] * other.q[2]
        t2 = self.q[0] * other.q[2] - \
             self.q[1] * other.q[3] + \
             self.q[2] * other.q[0] + \
             self.q[3] * other.q[1]
        t3 = self.q[0] * other.q[3] + \
             self.q[1] * other.q[2] - \
             self.q[2] * other.q[1] + \
             self.q[3] * other.q[0]
        q_prod = Quaternion(t0, ca.vertcat(t1, t2, t3))
        return q_prod

    @staticmethod
    def solve_quaternion_split(q, q_ref):
        quat_q = Quaternion(scalar=q[0], vec=q[1:4])
        quat_q_ref = Quaternion(scalar=q_ref[0], vec=q_ref[1:4])

        # Compute the quaternion error
        quat_error = quat_q_ref.__mul__(quat_q.inv())
        # print("\nSymbolic quaternion error (q_error):", quat_error.q)

        # Extract yaw (psi) from q_error
        psi = ca.atan2(2 * (quat_error.q[0] * quat_error.q[3] + quat_error.q[1] * quat_error.q[2]),
                       1 - 2 * (quat_error.q[2]**2 + quat_error.q[3]**2))

        # psi_func = ca.Function('psi_func', [q, q_ref], [psi])
        # psi_value = psi_func()
        # print("\nNumerical value of psi (yaw angle in radians):", psi_value)

        # Construct q_z from psi (rotation about z-axis [0,0,1])
        quat_z = Quaternion(scalar=ca.cos(psi/2), vec=ca.vertcat(0, 0, ca.sin(psi/2)))

        # Compute q_xy
        quat_z_inv = quat_z.inv()
        quat_xy = quat_z_inv.__mul__(quat_error)

        return quat_z.q, quat_xy.q

    def __str__(self):
        return f"Scalar: {self.q[0]}, Vector: {self.q[1:4]}"

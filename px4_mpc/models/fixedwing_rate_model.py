from acados_template import AcadosModel
import casadi as cs

def skew_symmetric(v):
    return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = cs.vertcat(
        cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)

    return cs.mtimes(rot_mat, v)

class FixedwingRateModel():
    def __init__(self):
        self.name = 'multirotor_rate_model'

        # constants
        self.mass = 1.
        self.max_thrust = 25
        self.max_rate = 6.8

    def get_acados_model(self) -> AcadosModel:

        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q = cs.MX.sym('q', 4)

        x = cs.vertcat(p, v, q)

        F = cs.MX.sym('F')
        w = cs.MX.sym('w', 3)
        u = cs.vertcat(F, w)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)

        xdot = cs.vertcat(p_dot, v_dot, q_dot)
        g = cs.vertcat(0.0, 0.0, -9.81) # gravity constant [m/s^2]
        
        force = cs.vertcat(0.0, 0.0, F)

        a_thrust = v_dot_q(force, q)/self.mass
        
        # dynamics
        f_expl = cs.vertcat(v,
                        a_thrust + g,
                        1 / 2 * cs.mtimes(skew_symmetric(w), q)
                        )

        f_impl = xdot - f_expl


        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model

from acados_template import AcadosModel
import casadi as cs

class FixedwingLongitudinalModel():
    def __init__(self):
        self.name = 'multirotor_rate_model'

        # constants
        self.mass = 1.
        self.max_thrust = 25
        self.max_rate = 6.8

    def get_acados_model(self) -> AcadosModel:

        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 2)
        v      = cs.MX.sym('v', 2)
        q = cs.MX.sym('q', 1)      # Pitch

        x = cs.vertcat(p, v, q)

        F = cs.MX.sym('F') # Thrust
        w = cs.MX.sym('w', 1) # pitch rate
        u = cs.vertcat(F, w)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 2)
        v_dot      = cs.MX.sym('v_dot', 2)
        q_dot      = cs.MX.sym('q_dot', 1)

        xdot = cs.vertcat(p_dot, v_dot, q_dot)
        g = cs.vertcat(0.0, -9.81) # gravity constant [m/s^2]
        
        force_thrust = cs.vertcat(F * cs.cos(q), F * cs.sin(q))
        density = 1.22
        area = 1
        Cl0 = 0.1
        Cla = 0.8
        Cd0 = 0.2
        Cda = 0.1
        dynamic_pressure = 0.5 * density * cs.norm_2(v) * cs.norm_2(v)
        theta = cs.atan(-v[1]/v[0])
        aoa = q - theta
        force_lift = (Cl0 + Cla * aoa)* area * dynamic_pressure
        force_drag = -(Cd0 + Cda * aoa) * area * dynamic_pressure
        force_aerodynamic = cs.vertcat(-force_lift * cs.sin(theta) + force_drag * cs.cos(theta), force_lift * cs.cos(theta) + force_drag * cs.sin(theta))

        acc = (force_thrust + force_aerodynamic)/self.mass + g
        
        # dynamics
        f_expl = cs.vertcat(v,
                        acc,
                        w
                        )

        f_impl = xdot - f_expl


        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model

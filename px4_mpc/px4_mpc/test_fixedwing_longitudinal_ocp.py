#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import numpy as np
from multirotor3d_model import plot_multirotor
from fixedwing_longitudinal_model import FixedwingLongitudinalModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as cs
import scipy.linalg
import matplotlib.pyplot as plt

def setup(model, x0, Fmax, wmax, N_horizon, Tf, use_RTI=False):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = 2*np.diag([0.0, 1e1, 1e-2, 1e-2, 0.0])
    Q_e = 2*np.diag([0.0, 1e1, 1e1, 1e1, 0.0])
    R_mat = 2*np.diag([1e-2, 1e-2])

    # TODO: How do you add terminal costs?

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    # ocp.cost.cost_type = 'EXTERNAL'
    # ocp.cost.cost_type_e = 'EXTERNAL'
    # ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
    # ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_e @ model.x

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_e


    ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.yref_e = np.array([0.0, 0.0, 10.0, 0.0, 0.0])

    # set constraints
    ocp.constraints.lbu = np.array([0.0, -wmax])
    ocp.constraints.ubu = np.array([+Fmax,  wmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-0.5])
    ocp.constraints.ubx = np.array([0.5])
    ocp.constraints.idxbx = np.array([4])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.print_level = 1
    if use_RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
    else:
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp.json')

    return ocp_solver, acados_integrator

def plot_fixedwing(shooting_nodes, f_max, w_max, U, X_true, Y_measured=None, latexify=False, X_true_label=None):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]

    num_plots = 5
    
    fig = plt.figure("Information", figsize=(8, 8))

    ax0 = fig.add_subplot(num_plots, 1, 1)
    line0, = ax0.step(t, np.append([U[0, 0]], U[:, 0]), label='F')
    ax0.hlines(f_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax0.hlines(0.0, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax0.set_ylim([0.0, 1.2*f_max])
    ax0.set_xlim(t[0], t[-1])
    ax0.legend(loc='lower right')

    ax1 = fig.add_subplot(num_plots, 1, 2)
    line1, = ax1.step(t, np.append([U[0, 1]], U[:, 1]), label='w_x')
    ax1.hlines(w_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax1.hlines(-w_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax1.set_ylim([-1.2*w_max, 1.2*w_max])
    ax1.set_xlim(t[0], t[-1])
    ax1.legend(loc='lower right')
    ax1.legend(loc='lower right')

    ax1.set_ylabel('$u$')
    ax1.set_xlabel('$t$')

    ax1.grid()

    ax2 = fig.add_subplot(num_plots, 1, 3)
    ax2.plot(t, X_true[:, 0], label='p_x')
    ax2.plot(t, X_true[:, 1], label='p_y')
    ax2.set_ylabel('X [m]')
    ax2.set_xlabel('$t$')
    ax2.grid()
    ax2.legend(loc=1)
    ax2.set_xlim(t[0], t[-1])

    ax3 = fig.add_subplot(num_plots, 1, 4)
    ax3.plot(t, X_true[:, 2], label='v_x')
    ax3.plot(t, X_true[:, 3], label='v_y')
    ax3.set_ylabel('V [m/s]')
    ax3.set_xlabel('$t$')
    ax3.grid()
    ax3.legend(loc=1)
    ax3.set_xlim(t[0], t[-1])    

    ax4 = fig.add_subplot(num_plots, 1, 5)
    ax4.plot(t, 180 * X_true[:, 4]/np.math.pi, label=r'$q_w$')
    ax4.set_ylabel(r'\theta')
    ax4.set_xlabel('$t$')
    ax4.grid()
    ax4.legend(loc=1)
    ax4.set_xlim(t[0], t[-1])

    plt.tight_layout()
    plt.show()


def main():

    N = 50
    model = FixedwingLongitudinalModel()
    acados_model = model.get_acados_model()

    Tf = 1.0
    nx = acados_model.x.size()[0]
    nu = acados_model.u.size()[0]

    Fmax = model.max_thrust
    wmax = model.max_rate

    x0 = np.array([0.1, 0.0, 10.0, 1.0, 0.0])
    ocp_solver, _ = setup(acados_model, x0, Fmax, wmax, N, Tf = 1.0, use_RTI=True)

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    plot_fixedwing(np.linspace(0, Tf, N+1), Fmax, wmax, simU, simX, latexify=False)


if __name__ == '__main__':
    main()

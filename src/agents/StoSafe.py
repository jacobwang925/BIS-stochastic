from .MobileAgent import MobileAgent
import numpy as np
from numpy.matlib import repmat
from numpy import zeros, eye, ones, sqrt, asscalar

from cvxopt import solvers, matrix

from qpsolvers import solve_qp

# import pydrake.solvers
# from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve

from datetime import datetime



class StoSafe(MobileAgent):
    """
    This is the Stochastic Safe Control method. Please refer to the paper for details.
    """

    # def __init__(self, d_min=2, k_v=2, yita=10):
    # def __init__(self, d_min=2):
    def __init__(self, d_min=2, alpha=1, epsilon=0.1, algo=1):

        MobileAgent.__init__(self);

        self.alpha = 0.5  # concave function
        self.H = 10  # outlook horizon

        self.half_plane_ABC = []

        self.safe_set = [0, 0, 0]
        self.alpha = alpha
        self.epsilon = epsilon
        self.algo = algo
        # self.k_v = k_v
        # self.d_min = d_min
        # self.yita = yita
        #
        # self.lambd = 0.5  # uncertainty margin

    def calc_control_input(self, dT, goal, fx, fu, Xr, Xh, dot_Xr, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, u0, min_u,
                           max_u, MC=True, F=None):

        # dim = np.shape(Mr)[0] // 2
        # p_idx = np.arange(dim)
        # v_idx = p_idx + dim
        #
        # d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])
        #
        # # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
        # # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)
        #
        # dot_Mr = p_Mr_p_Xr * dot_Xr
        # dot_Mh = p_Mh_p_Xh * dot_Xh
        #
        # dM = Mr - Mh
        # dot_dM = dot_Mr - dot_Mh
        # dp = dM[p_idx, 0]
        # dv = dM[v_idx, 0]
        #
        # dot_dp = dot_dM[p_idx, 0]
        # dot_dv = dot_dM[v_idx, 0]
        #
        # # dot_d is the component of velocity lies in the dp direction
        # dot_d = dp.T * dv / d
        #
        # p_dot_d_p_dp = dv / d - asscalar(dp.T * dv) * dp / (d ** 3)
        # p_dot_d_p_dv = dp / d
        #
        # p_dp_p_Mr = np.hstack([eye(dim), zeros((dim, dim))])
        # p_dp_p_Mh = -p_dp_p_Mr
        #
        # p_dv_p_Mr = np.hstack([zeros((dim, dim)), eye(dim)])
        # p_dv_p_Mh = -p_dv_p_Mr
        #
        # p_dot_d_p_Mr = p_dp_p_Mr.T * p_dot_d_p_dp + p_dv_p_Mr.T * p_dot_d_p_dv
        # p_dot_d_p_Mh = p_dp_p_Mh.T * p_dot_d_p_dp + p_dv_p_Mh.T * p_dot_d_p_dv
        #
        # p_dot_d_p_Xr = p_Mr_p_Xr.T * p_dot_d_p_Mr
        # p_dot_d_p_Xh = p_Mh_p_Xh.T * p_dot_d_p_Mh
        #
        # d = 1e-3 if d == 0 else d
        # dot_d = 1e-3 if dot_d == 0 else dot_d
        #
        # p_d_p_Mr = np.vstack([dp / d, zeros((dim, 1))])
        # p_d_p_Mh = np.vstack([-dp / d, zeros((dim, 1))])
        #
        # p_d_p_Xr = p_Mr_p_Xr.T * p_d_p_Mr
        # p_d_p_Xh = p_Mh_p_Xh.T * p_d_p_Mh
        #
        # phi = self.d_min ** 2 + self.yita * dT + self.lambd * dT - d ** 2 - self.k_v * dot_d;
        #
        # p_phi_p_Xr = - 2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr;
        # p_phi_p_Xh = - 2 * d * p_d_p_Xh - self.k_v * p_dot_d_p_Xh;
        #
        # dot_phi = p_phi_p_Xr.T * dot_Xr + p_phi_p_Xh.T * dot_Xh;
        #
        # L = p_phi_p_Xr.T * fu;
        # S = - self.yita - self.lambd - p_phi_p_Xh.T * dot_Xh - p_phi_p_Xr.T * fx;

        if self.algo == 1:
            # print('PotentialField')
            self.d_min = 2
            self.yita = 10
            self.lambd = 0.5
            self.k_v = 1
            # self.c1 = 3
            self.c1 = 5

            dim = np.shape(Mr)[0] // 2
            p_idx = np.arange(dim)
            v_idx = p_idx + dim

            d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])

            # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
            # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)

            dot_Mr = p_Mr_p_Xr * dot_Xr
            dot_Mh = p_Mh_p_Xh * dot_Xh

            dM = Mr - Mh
            dot_dM = dot_Mr - dot_Mh
            dp = dM[p_idx, 0]
            dv = dM[v_idx, 0]

            dot_dp = dot_dM[p_idx, 0]
            dot_dv = dot_dM[v_idx, 0]

            # dot_d is the component of velocity lies in the dp direction
            dot_d = dp.T * dv / d

            p_dot_d_p_dp = dv / d - asscalar(dp.T * dv) * dp / (d ** 3)
            p_dot_d_p_dv = dp / d

            p_dp_p_Mr = np.hstack([eye(dim), zeros((dim, dim))])
            p_dp_p_Mh = -p_dp_p_Mr

            p_dv_p_Mr = np.hstack([zeros((dim, dim)), eye(dim)])
            p_dv_p_Mh = -p_dv_p_Mr

            p_dot_d_p_Mr = p_dp_p_Mr.T * p_dot_d_p_dp + p_dv_p_Mr.T * p_dot_d_p_dv
            p_dot_d_p_Mh = p_dp_p_Mh.T * p_dot_d_p_dp + p_dv_p_Mh.T * p_dot_d_p_dv

            p_dot_d_p_Xr = p_Mr_p_Xr.T * p_dot_d_p_Mr
            p_dot_d_p_Xh = p_Mh_p_Xh.T * p_dot_d_p_Mh

            d = 1e-3 if d == 0 else d
            dot_d = 1e-3 if dot_d == 0 else dot_d

            p_d_p_Mr = np.vstack([dp / d, zeros((dim, 1))])
            p_d_p_Mh = np.vstack([-dp / d, zeros((dim, 1))])

            p_d_p_Xr = p_Mr_p_Xr.T * p_d_p_Mr
            p_d_p_Xh = p_Mh_p_Xh.T * p_d_p_Mh

            phi = self.d_min ** 2 + self.lambd * dT - d ** 2 - self.k_v * dot_d;
            p_dot_d_p_Mr = p_dp_p_Mr.T * p_dot_d_p_dp + p_dv_p_Mr.T * p_dot_d_p_dv
            p_d_p_Mr = np.vstack([dp / d, zeros((dim, 1))])
            p_phi_p_Mr = - 2 * d * p_d_p_Mr - self.k_v * p_dot_d_p_Mr

            u_Mr = -self.c1 * p_phi_p_Mr

            u_Xr = fu.T * p_Mr_p_Xr.T * u_Mr
            # print('u_Xr')
            # print(u_Xr)
            if phi > 0:
                u0 = u0 + u_Xr

        if self.algo == 2:
            self.lambd = 0.5  # uncertainty margin
            self.half_plane_ABC = []

            self.safe_set = [0, 0, 0]
            self.k_v = 0.2
            self.d_min = 1.5
            self.gamma = 5

            dim = np.shape(Mr)[0] // 2
            p_idx = np.arange(dim)
            v_idx = p_idx + dim

            d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])

            # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
            # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)

            dot_Mr = p_Mr_p_Xr * dot_Xr
            dot_Mh = p_Mh_p_Xh * dot_Xh

            dM = Mr - Mh
            dot_dM = dot_Mr - dot_Mh
            dp = dM[p_idx, 0]
            dv = dM[v_idx, 0]

            dot_dp = dot_dM[p_idx, 0]
            dot_dv = dot_dM[v_idx, 0]

            # dot_d is the component of velocity lies in the dp direction
            dot_d = dp.T * dv / d

            p_dot_d_p_dp = dv / d - asscalar(dp.T * dv) * dp / (d ** 3)
            p_dot_d_p_dv = dp / d

            p_dp_p_Mr = np.hstack([eye(dim), zeros((dim, dim))])
            p_dp_p_Mh = -p_dp_p_Mr

            p_dv_p_Mr = np.hstack([zeros((dim, dim)), eye(dim)])
            p_dv_p_Mh = -p_dv_p_Mr

            p_dot_d_p_Mr = p_dp_p_Mr.T * p_dot_d_p_dp + p_dv_p_Mr.T * p_dot_d_p_dv
            p_dot_d_p_Mh = p_dp_p_Mh.T * p_dot_d_p_dp + p_dv_p_Mh.T * p_dot_d_p_dv

            p_dot_d_p_Xr = p_Mr_p_Xr.T * p_dot_d_p_Mr
            p_dot_d_p_Xh = p_Mh_p_Xh.T * p_dot_d_p_Mh

            d = 1e-3 if d == 0 else d
            dot_d = 1e-3 if dot_d == 0 else dot_d

            p_d_p_Mr = np.vstack([dp / d, zeros((dim, 1))])
            p_d_p_Mh = np.vstack([-dp / d, zeros((dim, 1))])

            p_d_p_Xr = p_Mr_p_Xr.T * p_d_p_Mr
            p_d_p_Xh = p_Mh_p_Xh.T * p_d_p_Mh

            phi = self.d_min ** 2 + self.lambd * dT - d ** 2 - self.k_v * dot_d;

            p_phi_p_Xr = - 2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr;
            p_phi_p_Xh = - 2 * d * p_d_p_Xh - self.k_v * p_dot_d_p_Xh;

            dot_phi = p_phi_p_Xr.T * dot_Xr + p_phi_p_Xh.T * dot_Xh;

            L = p_phi_p_Xr.T * fu;
            S = - self.gamma * phi - p_phi_p_Xh.T * dot_Xh - p_phi_p_Xr.T * fx;

            u = u0;

            # if phi <= 0 or asscalar(L * u0) < asscalar(S):
            if asscalar(L * u0) < asscalar(S):
                u = u0;
            else:
                try:
                    # Q = matrix(eye(np.shape(u0)[0]))
                    # p = matrix(- 2 * u0)
                    # nu = np.shape(u0)[0]
                    # G = matrix(np.vstack([eye(nu), -eye(nu)]))
                    # r = matrix(np.vstack([max_u, -min_u]))
                    # A = matrix([[matrix(L),G]])
                    # b = matrix([[matrix(S),r]])
                    # solvers.options['show_progress'] = False
                    # sol=solvers.qp(Q, p, A, b)
                    # u = np.vstack(sol['x'])
                    u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
                except ValueError:
                    # print('no solution')
                    u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
                pass
                #

            u0 = u

            A = asscalar(L[0, 0]);
            B = asscalar(L[0, 1]);
            C = asscalar(S[0, 0]);

            self.half_plane_ABC = np.matrix([A, B, -C - A * Xr[0] - B * Xr[1]])
            self.ABC = np.matrix([A, B, C])

        else:
            # print('DAMN')
            pass

        if MC:
             u = u0

            # SCARA
            # if phi <= 0 or asscalar(L * u0) < asscalar(S):
            #     u = u0;
            # else:
            #     try:
            #         # Q = matrix(eye(np.shape(u0)[0]))
            #         # p = matrix(- 2 * u0)
            #         # nu = np.shape(u0)[0]
            #         # G = matrix(np.vstack([eye(nu), -eye(nu)]))
            #         # r = matrix(np.vstack([max_u, -min_u]))
            #         # A = matrix([[matrix(L),G]])
            #         # b = matrix([[matrix(S),r]])
            #         # solvers.options['show_progress'] = False
            #         # sol=solvers.qp(Q, p, A, b)
            #         # u = np.vstack(sol['x'])
            #         u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
            #     except ValueError:
            #         # print('no solution')
            #         u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
            #     pass
            # SCARA

            # A = asscalar(L[0,0]);
            # B = asscalar(L[0,1]);
            # C = asscalar(S[0,0]);
            #
            # self.half_plane_ABC = np.matrix([A, B, -C - A * Xr[0] - B * Xr[1]])
            # self.ABC = np.matrix([A, B, C])
            #
            # self.fuck = False

        else:
            dx = 0.1

            # alpha = 0.8
            # epsilon = 0.1

            dF = (F[0, :] - F[2, :]) / dx / 2

            print('F:')
            print(F)
            print('dF:')
            print(dF)

            u = u0

            # if all(dF > 0):
            if F[1, :].mean() > 1 - self.epsilon:
                # print('pass')
                pass
            else:
                # prog = MathematicalProgram()
                #
                # u_x = prog.NewContinuousVariables(2)
                #
                # u_y = prog.NewContinuousVariables(1)
                #
                # # G = -np.concatenate((Lg_F(x0, 0), [1])).reshape(1, 3)
                # # h = Lf_F(x0, 0) + alpha * (F(x0, 0) - (1 - epsilon))
                #
                # cost1 = prog.AddQuadraticErrorCost(Q=np.eye(2), x_desired=ctrl(0, x0), vars=u_x)
                #
                # cost2 = prog.AddQuadraticCost(Q=4000 * np.eye(1), b=np.zeros(1), c=0, vars=u_y)
                #
                # const = prog.AddLinearConstraint(
                #     A=A,
                #     lb=-np.inf * np.eye(1),
                #     ub=b * np.eye(1),
                #     vars=np.concatenate((u_x, u_y)))
                #
                # # G = -np.concatenate((Lg_F(x, k), [1])).reshape(1, 3)
                # # h = Lf_F(x, k) + alpha * (F_safe - (1 - epsilon))
                # cost1.evaluator().UpdateCoefficients(new_Q=np.eye(1), new_b=u0, new_c=0)
                # const.evaluator().UpdateCoefficients(new_A=A,
                #                                      new_lb=-np.inf * np.eye(1),
                #                                      new_ub=b)
                # u = Solve(prog).GetSolution()[:2]

                ## todo: QP solver
                Q = matrix(eye(np.shape(u0)[0]))
                # p = matrix(- 2 * u0)
                p = matrix(- u0)


                # print('fx:')
                # print(fx)

                A = dF * fu
                # b = -dF * fx * Xr + alpha * (F[1, 0] - (1-epsilon))
                b = -dF * fx + self.alpha * (F[1, :].mean() - (1 - self.epsilon))

                print('dF*fu: ')
                print(dF * fu)
                print(np.asarray(dF*fu).reshape(-1))

                ## soft constraint
                Q = matrix(eye(np.shape(u0)[0]+1))
                p = matrix(np.vstack([-u0, np.array(0)]))
                A = np.hstack([np.asarray(dF*fu).reshape(-1), np.array(-1)])
                b = -dF * fx + self.alpha * (F[1, :].mean() - (1 - self.epsilon))

                P = np.asarray(Q)
                # print(type(P)) # <class 'numpy.ndarray'>
                q = np.asarray(p).reshape(-1)
                G = np.asarray(A).reshape(-1)
                h = np.asarray(b).reshape(-1)


                print('P:')
                print(P)
                print('q:')
                print(q)
                print('G:')
                print(G)
                print('h:')
                print(h)

                Q = matrix(Q)
                p = matrix(p)

                A = matrix(A).T
                b = matrix(b)

                print('Q:')
                print(Q)
                print('p:')
                print(p)
                print('A:')
                print(A)
                print('b:')
                print(b)

                solvers.options['feastol'] = 1e-8
                solvers.options['show_progress'] = False

                x = solve_qp(P, q, G, h)

                print('x:')
                print(x)

                sol = solvers.qp(Q, p, A, b)
                # u = np.vstack(sol['x'])
                u = np.vstack(sol['x'][0:2])
                print('u:')
                print(u)
                print('sol:')
                print(sol['x'])
                # u = x
                # u = np.vstack(x)

                # now = datetime.now()
                # current_time = now.strftime("%H:%M:%S")
                # print("Current Time =", current_time)



        # print('Xr')
        # print(Xr.T)
        # print('Xh')
        # print(Xh.T)
        # print('dot Xr')
        # print(dot_Xr.T)
        # print('dot Xh')
        # print(dot_Xh.T)

        # print('p_phi_p_Xr')
        # print(p_phi_p_Xr.T)

        # print(- 2 * d * p_d_p_Xr)
        # print(- self.k_v * p_dot_d_p_Xr)

        # print('p_d_p_Xr')
        # print(p_d_p_Xr)

        # print('p_d_p_Mr')
        # print(p_d_p_Mr)

        # print('p_Mr_p_Xr')
        # print(p_Mr_p_Xr)

        # print('p_dot_d_p_Xr')
        # print(p_dot_d_p_Xr)

        # print('p_dot_d_p_Mr')
        # print(p_dot_d_p_Mr)

        # print('p_phi_p_Xh')
        # print(p_phi_p_Xh.T)
        # print(- 2 * d * p_d_p_Xh)
        # print(- self.k_v * p_dot_d_p_Xh)

        # print('p_d_p_Xh')
        # print(p_d_p_Xh)

        # print('p_dot_d_p_Xh')
        # print(p_dot_d_p_Xh)

        # print('phi')
        # print(phi)
        # print('dot_phi')
        # print(dot_phi)
        # print(p_phi_p_Xr.T * dot_Xr)
        # print(p_phi_p_Xh.T * dot_Xh)

        # print('d')
        # print(d)
        # print('dot_d')
        # print(dot_d);

        # print('L')
        # print(L)
        # print('fu')
        # print(fu)

        # print('S')
        # print(S)
        # print(- self.yita - self.lambd)
        # print(- p_phi_p_Xh.T * dot_Xh)
        # print(- p_phi_p_Xr.T * fx);
        # print('L * u0')
        # print(L * u0)
        # print('L * u')
        # print(L * u)
        # print('<')
        # print(asscalar(L * u0) < asscalar(S))

        # print('u0')
        # print(u0)
        # print('u')
        # print(u)
        self.fuck = False

        return u;

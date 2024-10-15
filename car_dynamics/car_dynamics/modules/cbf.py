import numpy as np
# import imageio
# import itertools
# import os
# import wandb
# import cvxpy as cp
# from scipy.optimize import minimize
import gymnasium as gym
from casadi import SX, vertcat, nlpsol, Function,horzcat, sum1
from .config import TrapType
from copy import deepcopy


class CBFController:
    def __init__(self, 
                 env: gym.Env,
                 params: dict,
                 hazardType: TrapType = TrapType.Hazard,
                 DEBUG:bool=False, 
                 ) -> None:
        self.env = env
        self.params = params
        self.DEBUG = DEBUG
        self.hazardType = hazardType
        
    def step(self, 
             input: dict,
             u_nominal: np.ndarray, 
        ):

        # barrier_fn = self.control_barrier_multi
        barrier_fn = self.control_barrier

        action_hat, info = barrier_fn(
                            input=input,
                            u_input=u_nominal,  
                        )
        return action_hat, info

    def control_barrier_multi(self, input, u_input):
        '''Calculate CBF for all hazards, solve joint qp'''


        heu_epsilon = self.params['heu_epsilon']
        alpha = self.params['alpha']
        kpos = self.params['kpos']
        kvel = self.params['kvel']

        state = input['current_state']
        curr_state = deepcopy(state)
        f_raw = input['f']
        g_raw = input['g']
        y_min = input['y_min']
        y_max = input['y_max']
        x_min = input['x_min']
        x_max = input['x_max']
        a_min = x_min[-2:]
        a_max = x_max[-2:]
        pos = state[:2]
        vel = state[2:]
        # print("vel: ", vel)

        hazard_pos_list = np.array(self.env.hazards_pos)[:,:2]
        hazard_size = self.env.hazards_size

        ### Kinematci CBF
        safeDistList = []
        for hazard_pos in hazard_pos_list:
            vec_pos = np.array([hazard_pos[0] - pos[0],
                                hazard_pos[1] - pos[1]])
            vel_proj = np.dot(vel, vec_pos) / np.linalg.norm(vec_pos)
            vel_norm = np.linalg.norm(vel_proj)
            ## Assume kvel be max acceleration
            safeDist = (hazard_size + heu_epsilon + vel_norm ** 2 / (kvel * 2) ) ** 2
            safeDistList.append(safeDist)


        # Define state variable (2-dimensional)
        x_t = SX(pos)
        u = SX.sym('u',2)
        # pos_list = SX(hazard_pos_list)
        pos_list = hazard_pos_list
        f = SX(f_raw) # dynamics parameter
        g = SX(g_raw)
        u_rl = SX(u_input) # known RL control input
        gamma = SX(alpha) # discount factor
        r = SX(hazard_size)  # radius parameter for obstacle
        # safe_dist = SX(heu_epsilon) # radius parameter for obstacle
        # safe_dist = SX(safeDist) # radius parameter for obstacle
        state = SX(state)
        # u_norm = (u_rl + u) * (a_max - a_min + 1e-8) + a_min
        u_norm = (u_rl + u - a_min) / (a_max - a_min + 1e-8)
        state_t_plus_1 = state + (f + (g @ u_norm)) * (y_max - y_min + 1e-8) + y_min
        x_t_plus_1 = state_t_plus_1[:2] 

        kh = 0
        # __import__('pdb').set_trace()
        h_x_0 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_0 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 1
        h_x_1 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_1 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 2 
        h_x_2 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_2 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 3 
        h_x_3 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_3 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 4 
        h_x_4 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_4 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 5 
        h_x_5 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_5 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 6 
        h_x_6 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_6 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        kh = 7 
        h_x_7 = (x_t[0] - pos_list[kh][0])**2 + (x_t[1] - pos_list[kh][1])**2 - safeDistList[kh] 
        h_next_7 = (x_t_plus_1[0] - pos_list[kh][0])**2 + (x_t_plus_1[1] - pos_list[kh][1])**2 - safeDistList[kh]

        # Define the constraint
        constraint0 = -h_next_0 + (1 - gamma) * h_x_0
        constraint1 = -h_next_1 + (1 - gamma) * h_x_1
        constraint2 = -h_next_2 + (1 - gamma) * h_x_2
        constraint3 = -h_next_3 + (1 - gamma) * h_x_3
        constraint4 = -h_next_4 + (1 - gamma) * h_x_4
        constraint5 = -h_next_5 + (1 - gamma) * h_x_5
        constraint6 = -h_next_6 + (1 - gamma) * h_x_6
        constraint7 = -h_next_7 + (1 - gamma) * h_x_7

        constraint_vel = u_rl + u
        constraints = vertcat(constraint0,
                              constraint1,
                              constraint2,
                              constraint3,
                              constraint4,
                              constraint5,
                              constraint6,
                              constraint7,
                              constraint_vel,
                            )

        # Define the objective function
        objective = sum1(u**2)

        # __import__('pdb').set_trace()
        # Define the optimization problem
        nlp = {'x': u,
               'f': objective, 
               # 'g': constraint, 
               'g': constraints, 
               # 'p': vertcat(x_t, pos, f, u_rl, gamma, r,safe_dist)
            }

        opts = {'ipopt.tol': 1e-3, 'print_time': 0, 'ipopt.print_level': 0}
        lbg = [
                -np.inf, #0
                -np.inf, #1 
                -np.inf, #2
                -np.inf, #3
                -np.inf, #4
                -np.inf, #5
                -np.inf, #6
                -np.inf, #7
                -1., 
                -1.
        ]
        ubg = [
                0., #0
                0., #1
                0., #2
                0., #3
                0., #4
                0., #5
                0., #6
                0., #7
                1., 
                1.,
        ]
        solver = nlpsol('solver', 'ipopt', nlp, opts)


        # Solve the QP
        solution = solver(x0=0, lbg=lbg,ubg=ubg)  # Change lbg to ubg

        # Print the solution
        # print("Optimal control input:", solution['x'])
        u_cbf = np.array(solution['x']).squeeze()
        u_out = (u_input + u_cbf)* (a_max - a_min + 1e-8) + a_min
        predictedNextPos = curr_state + (f_raw + np.dot(g_raw, u_out))* (y_max - y_min + 1e-8) + y_min
        # __import__('pdb').set_trace()
        return u_cbf, {'hazard_pos':hazard_pos, 'predictedNextPos': predictedNextPos}


    def control_barrier(self, input, u_input):


        heu_epsilon = self.params['heu_epsilon']
        alpha = self.params['alpha']
        kpos = self.params['kpos']
        kvel = self.params['kvel']

        state = input['current_state']
        curr_state = deepcopy(state)
        f_raw = input['f']
        g_raw = input['g']
        y_min = input['y_min']
        y_max = input['y_max']
        x_min = input['x_min']
        x_max = input['x_max']
        a_min = x_min[-2:]
        a_max = x_max[-2:]
        pos = state[:2]
        vel = state[2:]
        # print("vel: ", vel)


        if self.hazardType == TrapType.Hazard:
            hazard_pos = self.env.hazards_pos[0][:2]
            hazard_size = self.env.hazards_size
        elif self.hazardType == TrapType.Gremlin:
            hazard_pos = self.env.gremlins_locations[0][:2]
            hazard_size = self.env.gremlins_size
        elif self.hazardType == TrapType.MultiHazard:
            hazard_pos_list = np.array(self.env.hazards_pos)[:,:2]
            predictPos = pos + vel * 0.5
            dist_pos_list = np.linalg.norm(hazard_pos_list - predictPos, axis=1)
            minDistIndex = dist_pos_list.argmin()
            hazard_pos = hazard_pos_list[minDistIndex]
            hazard_size = self.env.hazards_size
            # print(f"[Info] Current pos -> {pos}\n  target hazard -> {hazard_pos}\n  whole list -> {hazard_pos_list}")

            # __import__('pdb').set_trace()


            
        else:
            raise Exception("Unknown Hazard Type!")


        ### Kinematci CBF
        vec_pos = np.array([hazard_pos[0] - pos[0],
                            hazard_pos[1] - pos[1]])
        vel_proj = np.dot(vel, vec_pos) / np.linalg.norm(vec_pos)
        vel_norm = np.linalg.norm(vel_proj)
        ## Assume kvel be max acceleration
        safeDist = (hazard_size + heu_epsilon + vel_norm ** 2 / (kvel * 2) ) ** 2


        # Define state variable (2-dimensional)
        x_t = SX(pos)
        u = SX.sym('u',2)
        pos = SX(hazard_pos)
        f = SX(f_raw) # dynamics parameter
        g = SX(g_raw)
        u_rl = SX(u_input) # known RL control input
        gamma = SX(alpha) # discount factor
        r = SX(hazard_size)  # radius parameter for obstacle
        # safe_dist = SX(heu_epsilon) # radius parameter for obstacle
        safe_dist = SX(safeDist) # radius parameter for obstacle
        state = SX(state)

        # h_xt = (x_t[0] - pos[0])**2 + (x_t[1] - pos[1])**2 - (r+safe_dist)**2
        h_xt = (x_t[0] - pos[0])**2 + (x_t[1] - pos[1])**2 - safe_dist

        u_norm = (u_rl + u) * (a_max - a_min + 1e-8) + a_min
        state_t_plus_1 = state + (f + (g @ u_norm)) * (y_max - y_min + 1e-8) + y_min

        # x_t_plus_1 = state_t_plus_1[:2] * kpos + state_t_plus_1[2:] * kvel
        x_t_plus_1 = state_t_plus_1[:2] 
        h_xt_plus_1 = (x_t_plus_1[0] - pos[0])**2 + (x_t_plus_1[1] - pos[1])**2 - safe_dist

        # Define the constraint
        constraint = -h_xt_plus_1 + (1 - gamma) * h_xt
        constraint_vel = u_rl + u
        constraints = vertcat(constraint, constraint_vel)

        # Define the objective function
        objective = sum1(u**2)

        # __import__('pdb').set_trace()
        # Define the optimization problem
        nlp = {'x': u,
               'f': objective, 
               # 'g': constraint, 
               'g': constraints, 
               # 'p': vertcat(x_t, pos, f, u_rl, gamma, r,safe_dist)
            }

        opts = {'ipopt.tol': 1e-3, 'print_time': 0, 'ipopt.print_level': 0}
        lbg = [-np.inf, -1., -1.]
        ubg = [0., 1., 1.]
        solver = nlpsol('solver', 'ipopt', nlp, opts)

        # # Define parameters for your problem
        # params = {
        #     'x_t': pos,  # Current state
        #     'pos': hazard_pos,  # Obstacle position
        #     'f': f_raw,    # Dynamics parameter
        #     # 'g': g_raw,    # Control parameter
        #     'u_rl': u_input, # Known RL control input
        #     'gamma': alpha,       # Discount factor
        #     'r': hazard_size,            # Radius parameter for obstacle
        #     'safe_dist': heu_epsilon,
        # }

        # Solve the QP
        solution = solver(x0=0, lbg=lbg,ubg=ubg)  # Change lbg to ubg

        # Print the solution
        # print("Optimal control input:", solution['x'])
        u_cbf = np.array(solution['x']).squeeze()
        u_out = (u_input + u_cbf)* (a_max - a_min + 1e-8) + a_min
        predictedNextPos = curr_state + (f_raw + np.dot(g_raw, u_out))* (y_max - y_min + 1e-8) + y_min
        # __import__('pdb').set_trace()
        return u_cbf, {'hazard_pos':hazard_pos, 'predictedNextPos': predictedNextPos}



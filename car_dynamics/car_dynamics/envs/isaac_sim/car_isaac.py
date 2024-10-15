import os
import gym
from gym import spaces
from copy import deepcopy
import numpy as np
import math
import yaml
from scipy.spatial.transform import Rotation as R

import sys
import matplotlib.pyplot as plt
from car_dynamics import ISAAC_ASSETS_DIR

#Needs to be run with /insert path/isaac-sim-2023.1.1/python.sh

class IsaacCar(gym.Env):
    DEFAULT = {
        'max_step': 100,
        'dt': 0.02,
        'is_render': False,
        'delay': 0,
        'warmup_steps': 40,
        'usd_name': "F1Tenth_lecar.usd",
        'max_throttle': 0.05,
        'max_steer': 0.3,
        'steer_bias': 0
    }
    print("here")
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "fast_shutdown": True}) # we can also run as headless.
    simulation_app.update()
    simulation_app.update()

    @property
    def usd_path(self):
        return os.path.join(ISAAC_ASSETS_DIR, self.usd_name)
    
    @property
    def cone_usd_path(self):
        return os.path.join(ISAAC_ASSETS_DIR, "omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd")

    def __init__(self, config: dict):
        super(IsaacCar, self).__init__()
        
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
        
        self.action_space = spaces.Box(low=np.array([-1., -1.]), 
                                       high=np.array([1., 1.]), dtype=np.float32)
               
        self.observation_space = spaces.Box(
                ## x, y, psi, vx, vy, omega
                low=np.array([-np.inf] * 6), 
                high=np.array([np.inf] * 6), 
                dtype=np.float32,
        )

        self.hz = 50
        self.sim, self.world, self.ctrl = self.initialize_simulation(self.simulation_app)

        self._step = None
        
        self.action_buffer = []

        self.name = "isaac"

        #track_width = 24
        #wheelbase = 32
        
        self.wheelbase = 0.32 #32 centimeters

        self.reset()
        
    def obs_state(self):
        [x, y, z, ori_w, ori_x, ori_y, ori_z, vx, vy, vz, ax, ay, az, wx, wy, wz ] = self.full_obs
        
        r = R.from_quat([ori_x, ori_y, ori_z, ori_w])
        yaw = r.as_euler('zyx')[0]
        
        return np.array([x, #pos x 
                         y, #pos y
                         yaw, #yaw
                         vx, #lin x
                         vy, #lin y
                         wz]) #angular z

    def reset(self):
        self.world.reset()

        self._step = 0
        self.action_buffer = []
        # self.warmup_sim()
            
        print("reset called")
            
        for _ in range(self.delay):
            self.action_buffer.append(np.array([0., 0.], dtype=np.float32))
            
        return self.obs_state()
    
    def reward(self):
        return .0
    
    #TODO change update frequency from 60hz to 50hz
    def step(self, action_):
        from omni.isaac.core.utils.types import ArticulationAction
       
        action_ = np.array(action_, dtype=np.float32)
        self.action_buffer.append(action_)
   
        action = self.action_buffer[0].copy()
        assert action.dtype == np.float32
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # num_steps = int(self.dt/self.timescale)

        # convert action to true action in isaac sim
        action_isaac = np.array([action[0] * self.max_throttle, action[1]*self.max_steer + self.steer_bias], dtype=np.float32)
        
        #apply the actionn
        RL = self.ctrl.get_dof_index("Wheel__Upright__Rear_Left")
        RR = self.ctrl.get_dof_index("Wheel__Upright__Rear_Right")
        FL = self.ctrl.get_dof_index("Wheel__Knuckle__Front_Left")
        FR = self.ctrl.get_dof_index("Wheel__Knuckle__Front_Right")

        LSTR = self.ctrl.get_dof_index("Knuckle__Upright__Front_Left")
        RSTR = self.ctrl.get_dof_index("Knuckle__Upright__Front_Right")

        throttle = ArticulationAction(joint_efforts=np.array([action_isaac[0], action_isaac[0], action_isaac[0], action_isaac[0]]), joint_indices=np.array([FL, FR, RL, RR]))
        self.ctrl.apply_action(throttle)
        # self.ctrl.get_articulation_controller().apply_action(throttle)
        # str_cmd = cmd[i]
        steer = ArticulationAction(joint_positions=np.array([action_isaac[1], action_isaac[1]]), joint_indices=np.array([LSTR, RSTR]))
        self.ctrl.apply_action(steer)
        # self.ctrl.get_articulation_controller().apply_action(steer)

        # step twice since sim is running at 100hz
        for _ in range(self.hz//50):
            self.world.step(render = self.is_render)     
            self._step += 1

        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False
        
        self.action_buffer.pop(0)
        
        return self.obs_state(), reward, done, {}
    
    def change_parameters(self, parameters, change = True):
        if change:
            print("Changing Parameters...s")
            for key, item in parameters.items():
                if key == "mass":
                    self.ctrl.set_body_masses(np.array([item]), indices = np.array([0]), body_indices = np.array([0]))
                elif key == "com":
                    positions = np.array([[item]])
                    orientations = np.array([[[1.0, 0.0, 0.0, 0.0]]])
                    
                    self.ctrl.set_body_coms(positions, orientations, indices = np.array([0]), body_indices = np.array([0]))
                elif key == "friction":
                    self.pm.set_dynamic_friction(item)
                    self.pm.set_static_friction(item)
                elif key == "max_throttle":
                    self.max_throttle = item
                elif key == "max_steer":
                    self.max_steer = item
                elif key == "steer_bias":
                    self.steer_bias = item
                elif key == "wheelbase" or "sim" or "delay":
                    pass
                else:
                    print("Invalid parameter: ", key)
        else:
            print("[WARN] Not Changing Parameters")
            

    def shutdown(self):
        print("[Isaac Sim] Shutdown does not currently work as intended")
        self.simulation_app.close()

    # generate a new mass
    def generate_new_mass(self):   
        # print("[Warn] Car Mass Generation Not Defined for Simulator Type")
        # mass_dict = {}
        # for i, name in enumerate(self.ctrl.body_names):
        #     mass_dict[name] = self.ct add_reference_to_stage(self.usd_path, "/World/F1Tenth")rl.get_body_masses()[0][i]
        # print("Mass Dict", mass_dict)

        default_mass =  3.0 # base isaacsim mass
        
        lower = default_mass * 0.7
        upper = default_mass * 1.3
        new_mass = np.random.uniform(lower, upper)

        return new_mass

    def generate_new_com(self):
        # print("[Warn] COM Generation Not Defined for Simulator Type")
        # print("Body COMs", self.ctrl.get_body_coms())
        # com_dict = {}
        # for i, name in enumerate(self.ctrl.body_names):
        #     com_dict[name] = self.ctrl.get_body_coms()[1][0][i]
        # print("COM Dict", com_dict)
    
        default_com = np.array([-1.0389194e-02, -7.5846906e-08,  2.2499999e-02]) # base isaacsim COM, 
        #orientation [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.1056628e-06]
        lower = default_com - 0.05 #5 cm range
        upper = default_com + 0.05

        new_com = np.random.uniform(lower, upper)

        return new_com
    
    def generate_new_friction(self):
        # print("[Warn] Friction Generation Not Defined for Simulator Type")
        # print("Joint Frictions", self.ctrl.get_friction_coefficients())
        default_friction = 0.9 #base isaacsim friction
        lower = default_friction * 0.5
        upper = default_friction * 1.1
        new_friction = np.random.uniform(lower, upper)
        
        return new_friction

    def generate_new_delay(self):
        lower = 0
        upper = 6
        new_delay = int(np.random.uniform(lower, upper))
        return new_delay
   
    def generate_new_max_throttle(self):
        # print("[Warn] Max Throttle Generation Not Defined for Simulator Type")
        lower = 0.01
        upper = 0.1
        max_thr = np.random.uniform(lower, upper)
        return 0.1 #max_thr
    
    def generate_new_max_steering(self):
        # print("[Warn] Max Steering Generation Not Defined for Simulator Type")
        lower = 0.15
        upper = 0.36
        max_steer = np.random.uniform(lower, upper)
        return max_steer
    
    def generate_new_steering_bias(self):
        lower = 0.0
        upper = 0.01
        bias = np.random.uniform(lower, upper)
        return bias
    
    def generate_new_slope(self):
        # TODO not implemented right now
        print("[Warn] Slope Generation Not Defined for Simulator Type")
        pass
    

    @property
    def full_obs(self): 
        pose, orientation = self.ctrl.get_world_poses() #xyz, wxyz
        lin_vel = self.ctrl.get_linear_velocities()
        a_vel = self.ctrl.get_angular_velocities()
        
        pose = pose[0]
        orientation = orientation[0]
        lin_vel = lin_vel[0]
        a_vel = a_vel[0]

        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2])
        
        ori_w = float(orientation[0])
        ori_x = float(orientation[1])
        ori_y = float(orientation[2])
        ori_z = float(orientation[3])
        
        ori_w = float(orientation[0])
        ori_x = float(orientation[1])
        ori_y = float(orientation[2])
        ori_z = float(orientation[3])
        
        yaw = R.from_quat(np.array([ori_x, ori_y, ori_z, ori_w])).as_euler("xyz")[2]
        
        H = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                      [np.sin(-yaw), np.cos(-yaw), 0],
                      [0, 0, 1]])
        
        transformed_vs = H @ np.array([lin_vel[0], lin_vel[1], 1])
        
        vx = float(transformed_vs[0])
        vy = float(transformed_vs[1])
        vz = float(lin_vel[2])
        
        
        wx = float(a_vel[0])
        wy = float(a_vel[1])
        wz = float(a_vel[2])
        
        # no accel information
        ax = float(0)
        ay = float(0)
        az = float(0)
    
        return np.array([x, y, z, ori_w, ori_x, ori_y, ori_z, vx, vy, vz, ax, ay, az, wx, wy, wz ])

    @property
    def pose(self):
        pose, orientation = self.ctrl.get_world_poses() #xyz, wxyz
        pose = pose[0]
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2])
        return np.array([x, y ,z])

    @property 
    def rpy(self):
        pose, orientation = self.ctrl.get_world_poses() #xyz, wxyz
        orientation = orientation[0]
        ori_w = float(orientation[0])
        ori_x = float(orientation[1])
        ori_y = float(orientation[2])
        ori_z = float(orientation[3])
        rpy = R.from_quat(np.array([ori_x, ori_y, ori_z, ori_w])).as_euler("xyz")
        return rpy
    
    def spawn_track(self, track):
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
        from omni.isaac.core.prims import XFormPrim
        num_cones = 20
        cone_interval = track.shape[0] // num_cones
        track_width = 1.5
        start_pos = track[0]
        orient = R.from_euler("xyz", [0, 0, 0]).as_quat()
        self.ctrl.set_world_poses(positions = np.array([[start_pos[0], start_pos[1], 0]])/ get_stage_units(), 
                                  orientations=np.array([[orient[3], orient[0], orient[1], orient[2]]]) / get_stage_units()) #wxyz
        # print("cone interval",cone_interval)
        for i in range(num_cones):
            cx, cy = track[i * cone_interval]
            cx_next, cy_next = track[i*cone_interval + 1]
            a = cx_next - cx
            b = cy_next - cy
            mid_x = (cx+ cx_next) / 2
            mid_y = (cy + cy_next) / 2
            ortho_vec = np.array([-b, a])
            norm_ortho_vec = ortho_vec/np.linalg.norm(ortho_vec)
            cone1_pos = track_width/2 * norm_ortho_vec + np.array([mid_x, mid_y])
            cone2_pos = -1 * track_width/2 * norm_ortho_vec + np.array([mid_x, mid_y])
            
            primname1 = f"/World/cones/cone{i}_1"
            primname2 = f"/World/cones/cone{i}_2"
            print(f"adding cone{i} pair to scene")
            add_reference_to_stage(self.cone_usd_path, primname1)
            cone1 = self.world.scene.add(XFormPrim(prim_path=primname1, 
                                                  name=f"cone{i}_1",
                                                  position=np.array([[cone1_pos[0], cone1_pos[1], 0]]), 
                                                  scale = np.array([0.5, 0.5, 0.5])))
            add_reference_to_stage(self.cone_usd_path, primname2)
            cone2 = self.world.scene.add(XFormPrim(prim_path=primname2, 
                                                  name=f"cone{i}_2",
                                                  position=np.array([[cone2_pos[0], cone2_pos[1], 0]]), 
                                                  scale = np.array([0.5, 0.5, 0.5])))
        

    def initialize_simulation(self, simulation_app):
        from omni.isaac.core import World
        from omni.isaac.core.objects import GroundPlane
        from omni.isaac.core.robots import RobotView
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.prims import GeometryPrim
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
        from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
        from omni.isaac.core.utils.stage import is_stage_loading
        import carb
    
        world = World()
        
        #change simulation timesteps here
        world.set_simulation_dt(physics_dt=1.0 / self.hz, rendering_dt=1.0 / 50.0)

        # make sure the file exists before we try to open it

        simulation_app.update

        try:
            result = is_file(self.usd_path)
        except:
            result = False

        if result:
            # omni.usd.get_context().open_stage(usd_path)
            add_reference_to_stage(self.usd_path, "/World/F1Tenth")
            art_system = world.scene.add(RobotView(prim_paths_expr='/World/F1Tenth', name="ihateisaacsim"))
        else:
            carb.log_error(
                f"the usd path {self.usd_path} could not be opened"
            )
            simulation_app.close()
            sys.exit()

        print("Loading stage...")
        while is_stage_loading():
            simulation_app.update()

        print("Loading Complete")
        
        self.pm = PhysicsMaterial(prim_path = "/World/F1Tenth/Rubber_Asphalt")
        # self.pm.set_dynamic_friction(0.00)
        # self.pm.set_static_friction(0.00)
        plane = GroundPlane(prim_path="/World/GroundPlane", z_position=0)
        plane.apply_physics_material(self.pm)
        world.scene.add(plane)
    
        FL_prim = GeometryPrim("/World/F1Tenth/Rigid_Bodies/Wheel_Front_Left")
        FR_prim = GeometryPrim("/World/F1Tenth/Rigid_Bodies/Wheel_Front_Right")
        RL_prim = GeometryPrim("/World/F1Tenth/Rigid_Bodies/Wheel_Rear_Left")
        RR_prim = GeometryPrim("/World/F1Tenth/Rigid_Bodies/Wheel_Rear_Right")
        FL_prim.apply_physics_material(self.pm)
        FR_prim.apply_physics_material(self.pm)
        RL_prim.apply_physics_material(self.pm)
        RR_prim.apply_physics_material(self.pm)

        world.initialize_physics()

        world.play()

        return simulation_app, world, art_system
    
    def warmup_sim(self):
        warmupsteps = 200
        for _ in range(warmupsteps):
            self.world.step(render=True)

    
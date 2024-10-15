import os
import pickle
import numpy as np
import mujoco as mj
from copy import deepcopy
from mujoco.glfw import glfw
from .cam_utils import update_camera
from scipy.spatial.transform import Rotation as R


def initialize_environment(xml_path, timestep, render):
    
    # For callback functions
    global button_left 
    global button_middle
    global button_right
    global lastx
    global lasty

    #get the full path
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)                # MuJoCo data   
    cam = mj.MjvCamera()                        # Abstract camera
    opt = mj.MjvOption() 
    model.opt.timestep = timestep

    if render:  # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        window = glfw.create_window(1000, 600, "Demo", None, None) # visualization options  
        glfw.make_context_current(window)
        glfw.swap_interval(1)   
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)     
    else:
        scene = None
        context = None
        window = None
                         

    # print("Model Update Rate in Hz: ", 1/model.opt.timestep)
    # print("Model Geoms: ", [model.geom(i).name for i in range(model.ngeom)])
    # print("Body Geoms: ", [model.body(i).name for i in range(model.nbody)])

    return model, data, cam, opt, scene, context, window


class World:
    
    DEFAULT = {
        'timestep': 0.001,
        'is_render': False,
        'xml_path': 'models/one_turbo_slope.xml',
        'max_throttle': 8.0,
        'max_steer': 0.36,
        'steer_bias': 0
    }
    
    def __init__(self, config={}):
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
            

        model, data, cam, opt, scene, context, window  = initialize_environment(self.xml_path, self.timestep, self.is_render)
        
        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scene = scene
        self.context = context
        self.window = window
        
        self.dt = self.model.opt.timestep
        self.warmup_steps = int(2. / self.timestep)
        
        
    def reset(self, ):
        # self.model.body_pos[1] = np.hstack((trajectory[0],[0]))
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        for _ in range(self.warmup_steps):
            self.data.ctrl[1] = 0.
            self.data.ctrl[0] = 0.
            mj.mj_step(self.model, self.data)
            # print("height", self.pose[2])
        
    def step(self, actions):
        """Step the simulation forward by one timestep
            map controller output [-1, 1]
            - to [0, max_speed] for throttle
            - to [-max_steering, max_steering] for steering

        Args:
            actions (_type_): _description_
        """
        # self.data.ctrl[1] = actions[0] * (self.max_speed / 2) + (self.max_speed / 2)
        self.data.ctrl[1] = actions[0] # mapping throttle to correct control
        # self.data.ctrl[0] = actions[1] * self.max_steering
        self.data.ctrl[0] = actions[1] # mapping steering to correct control
        mj.mj_step(self.model, self.data)
        
    def render(self, mode='human'):
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        update_camera(self.cam, self.data.geom_xpos[1])
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
            
    @property
    def pose(self):
        """global position [x, y, z] of the car

        Returns:
            np.array: [x, y, z]
        """
        return np.array(
            [
                self.data.geom_xpos[1][0],
                self.data.geom_xpos[1][1],
                self.data.geom_xpos[1][2],
            ]
        ).copy()
    
    @property
    def orientation(self):
        """global orientation [w, x, y, z] of the car
        
        Returns:
            np.array: [w, x, y, z]
        """
        orientation = R.from_matrix(self.data.geom_xmat[2].reshape((3,3))).as_quat()
        return np.array(
            [
                orientation[3],
                orientation[0],
                orientation[1],
                orientation[2],
            ]
        ).copy()
    
    @property
    def rpy(self):
        """global orientation [x, y, z] of the car
        
        Returns:
            np.array: [x, y, z]
        """
        orientation = R.from_matrix(self.data.geom_xmat[2].reshape((3,3))).as_euler("xyz")
        return np.array(
            [
                orientation[0],
                orientation[1],
                orientation[2],
            ]
        ).copy()
        
    @property
    def lin_vel(self):
        """linear velocity [vx, vy, vz] of the car
        
        Returns:
            np.array: [vx, vy, vz]
        """
        return np.array(
            [
                self.data.sensor('velocimeter').data[0],
                self.data.sensor('velocimeter').data[1],
                self.data.sensor('velocimeter').data[2],
            ]
        ).copy()
        
    @property
    def ang_vel(self):
        """angular velocity [wx, wy, wz] of the car
        
        Returns:
            np.array: [wx, wy, wz]
        """
        return np.array(
            [
                self.data.sensor('gyro').data[0],
                self.data.sensor('gyro').data[1],
                self.data.sensor('gyro').data[2],
            ]
        ).copy()
        
    @property
    def lin_acc(self):
        """linear acceleration [ax, ay, az] of the car
        
        Returns:
            np.array: [ax, ay, az]
        """
        return np.array(
            [
                self.data.sensor('accelerometer').data[0],
                self.data.sensor('accelerometer').data[1],
                self.data.sensor('accelerometer').data[2],
            ]
        ).copy()
    
    def change_parameters(self, parameters, change = True):
        if change:
            for key, item in parameters.items():
                if key == "mass":
                    self.model.body_mass[1] = item
                elif key == "com":
                    self.model.body_ipos[1] = item
                elif key == "friction":
                    self.model.geom_friction[5] = item
                    self.model.geom_friction[7] = item
                    self.model.geom_friction[9] = item
                    self.model.geom_friction[11] = item
                    self.model.geom_friction[0] = item
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
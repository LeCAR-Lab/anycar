#launch Isaac Sim before any other imports

#default first two lines in any standalone application

#/home/tony/.local/share/ov/pkg/isaac-sim-2023.1.1/python.sh

import sys
import matplotlib.pyplot as plt

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import is_stage_loading

import carb
import numpy as np

simulation_app.update()
simulation_app.update()

simulation_app.close()

world = World()
world.scene.add_default_ground_plane()
world.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

usd_path = "assets/F1Tenth.usd"

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    # omni.usd.get_context().open_stage(usd_path)
    add_reference_to_stage(usd_path, "/F1Tenth")
    art_system = world.scene.add(Robot(prim_path="/F1Tenth", name="f1_tenth"))
    art_system.set_world_pose(position = np.array([0, 0, 0])/ get_stage_units(), orientation=np.array([1, 0, 0,  0]) / get_stage_units()) #wxyz

    print("world scale", art_system.get_world_scale())
    print("stage units", get_stage_units())
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened"
    )
    simulation_app.close()
    sys.exit()

print("Loading stage...")
while is_stage_loading():
    simulation_app.update()

print("Loading Complete")

world.initialize_physics()


print("DOF names", art_system.dof_names)

RL = art_system.get_dof_index("Wheel__Upright__Rear_Left")
RR = art_system.get_dof_index("Wheel__Upright__Rear_Right")
FL = art_system.get_dof_index("Wheel__Knuckle__Front_Left")
FR = art_system.get_dof_index("Wheel__Knuckle__Front_Right")

LSTR = art_system.get_dof_index("Knuckle__Upright__Front_Left")
RSTR = art_system.get_dof_index("Knuckle__Upright__Front_Right")

print(art_system.dof_properties["stiffness"][LSTR])
print(art_system.dof_properties["stiffness"][RSTR])

world.play()
steer_pos = []
steer_vel = []
pose = []
vel = []
cmd = np.linspace(-0.4, 0.4, 1000)
# art_system.set_world_pose(orientation=np.array([0.7068252, 0, 0,  0.7073883]) / get_stage_units()) #wxyz
# 

# world.initialize_physics()

warmupsteps = 200
for i in range(warmupsteps):
    world.step(render=True)

for i in range(1000):
    # Run in realtime mode, we don't specify the step size
    throttle = ArticulationAction(joint_efforts=np.array([0.003, 0.003, 0.003, 0.003]), joint_indices=np.array([FL, FR, RL, RR]))
    art_system.get_articulation_controller().apply_action(throttle)
    # str_cmd = cmd[i]
    steer = ArticulationAction(joint_positions=np.array([0.0, 0.0]), joint_indices=np.array([LSTR, RSTR]))
    art_system.get_articulation_controller().apply_action(steer)
    steer_pos.append(art_system.get_joint_positions(joint_indices=np.array([LSTR, RSTR])))
    steer_vel.append(art_system.get_joint_velocities(joint_indices=np.array([LSTR, RSTR])))
    pose.append(art_system.get_world_pose()[0])
    vel.append(art_system.get_linear_velocity())
    world.step(render=True)
    # input()

print("Sim Complete!")

# world.reset()
# warmupsteps = 200
# for i in range(warmupsteps):
#     world.step(render=True)

# for i in range(1000):
#     # Run in realtime mode, we don't specify the step size
#     throttle = ArticulationAction(joint_efforts=np.array([-0.01, -0.01, -0.01, -0.01]), joint_indices=np.array([FL, FR, RL, RR]))
#     art_system.get_articulation_controller().apply_action(throttle)
#     # str_cmd = cmd[i]
#     steer = ArticulationAction(joint_positions=np.array([0.3, 0.3]), joint_indices=np.array([LSTR, RSTR]))
#     art_system.get_articulation_controller().apply_action(steer)
#     steer_pos.append(art_system.get_joint_positions(joint_indices=np.array([LSTR, RSTR])))
#     steer_vel.append(art_system.get_joint_velocities(joint_indices=np.array([LSTR, RSTR])))
#     pose.append(art_system.get_world_pose()[0])
#     world.step(render=True)
#     # input()


# print(pose)

pose = np.array(pose)
steer_pos = np.array(steer_pos)
steer_vel = np.array(steer_vel)
vel = np.array(vel)
plt.plot(steer_pos[:, 0], label = "leftpos")
plt.plot(steer_pos[:, 1], label = "rightpos")
plt.plot(steer_vel[:, 0], label = "leftvel")
plt.plot(steer_vel[:, 1], label = "rightvel")
plt.legend()
plt.show()


plt.plot(pose[:, 0], pose[:, 1])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plt.clf()
plt.plot(vel[:, 0], label = "xvel")
plt.plot(vel[:, 1], label = "yvel")
plt.legend()
plt.show()
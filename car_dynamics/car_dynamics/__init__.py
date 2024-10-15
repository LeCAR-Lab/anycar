import os

CAR_DYNAMICS_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

MUJOCO_MODEL_DIR = os.path.join(CAR_DYNAMICS_ROOT_DIR, "car_dynamics", "envs", "mujoco_sim", "models")
UNITY_ASSETS_DIR = os.path.join(CAR_DYNAMICS_ROOT_DIR, "car_dynamics", "envs", "unity_sim",)
ISAAC_ASSETS_DIR = os.path.join(CAR_DYNAMICS_ROOT_DIR, "car_dynamics", "envs", "isaac_sim", "assets")
ASSETTO_CORSA_ASSETS_DIR = os.path.join(CAR_DYNAMICS_ROOT_DIR, "car_dynamics", "envs", "assetto_corsa")

TMP_DIR = os.path.join(CAR_DYNAMICS_ROOT_DIR, "tmp")

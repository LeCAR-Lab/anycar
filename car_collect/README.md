# Toolbox for data collection in different platform

## `assetto_corsa_collect` : For Assetto Corsa Gym

   - `acc_deploy_mppi_tcp.py` Connects with the Assetto Corsa Environment and listens to commands via TCP
   - `acc_mppi_client.py` Runs MPPI and communicates with `acc_deploy_mppi_tcp.py` via TCP
   - `acc_manual_client.py` Allows for manual commands
   - `acc_mppi_client_ros.py` Allows for MPPI to run on separate `car_node.py` and relays ros messages to `acc_deploy_mppi_tcp.py` via TCP
   - `acc_pure_pursuit.py` Standalone program that allows deployment of pure pursuit directly.


## `isaacsim_collect`: For IsaacSim

> Note: To run isaac sim, you need to activate the conda environment first.
> ```bash
>  source /home/lecar/.local/share/ov/pkg/isaac-sim-2023.1.1/setup_conda_env.sh
> ```
> 

   - `isaacsim_collect_data.py` Creates Isaacsim Environment and collects data
   - `isaacsim_on_policy.py` Creates the Isaacsim Environment and listens to commands via TCP
   - `isaacsim_client_ros.py` Allows for MPPI to run on separate `car_node.py` and relays ros messages to `isaacsim_on_policy.py` via TCP
   - `isaacsim_mppi_client.py` Runs MPPI and communicates with `isaacsim_on_policy.py` via TCP

## `mujoco_collect` : For MuJoCo sim
   
   - `parallel_mujoco_collect_data.py` Creates MuJoCo Environment and collects data

## `numeric_collect`: For Numerical DBM sim

   - `collect_data_on_policy.py`  Creates numeric sim environments and deploys MPPI to collect on-policy data
   - `collect_data_gym.py` Creats numeric sim environments
---


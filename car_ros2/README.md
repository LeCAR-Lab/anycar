
# Instruction For Real-World Deployment
## File Tree
- setup

```
source install/setup.bash 
```

- run foxglove node:

```
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

- run car_sim node:

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py
```

- real2sim node:

```
ros2 launch car_ros2 real2sim_node.launch.py 
```


## Customize Parameters

In `car_ros2/utils.py`

```python
def load_mppi_params() -> MPPIParams:
    return MPPIParams(
        spline_order=2,
        sigma=0.05,
        gamma_sigma=0.0,
        gamma_mean=1.0,
        discount=1.0,
        sample_sigma=1.0,
        lam=0.1,
        n_rollouts=600,        
        a_min=[-1, -1.], # first dim steer, 2nd throttle
        a_max=[1., 1.],
        a_mag=[1., 1.],
        a_shift=[0., 0.],
        delay=0,
        len_history=251,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        num_intermediate=7,
        h_knot=8,
        smooth_alpha=1.0,
        dynamics="transformer-jax",
        dual=True, 
        # dynamics="dbm",
        # dual=False, 
    )
```

## Deploy in Real

1. Use nomachine to connect to car:

2. In Desktop:

    - read Joystick signal: `ros2 run joy_linux joy_linux_node --ros-args -p dev:=/dev/input/js0`
    - run Joystick node: `ros2 run carros2 car_joystick_node`
    - steer the car
    - turn off joystick
    - run car_node `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 ros2 launch car_ros2 car_real.launch.py`
    - run foxglove bridge: `ros2 launch foxglove_bridge foxglove_bridge_launch.xml `


3. Open 3 separate terminals

- Terminal 1:

    ```bash
    source install/setup.bash
    source ~/f1tenth/install/setup.bash
    ros2 launch lecar_launch vesc.launch.py
    ```

- Terminal 2 (For Lidar SLAM):

    ```bash
    source install/setup.bash
    source ~/f1tenth/install/setup.bash
    ros2 launch art_f1tenth_interface art_f1tenth_interface.launch.py
    ros2 launch lecar_launch lidar.launch.py
    ```

- Terminal 2 (For VIO):

    ```bash
    source install/setup.bash
    ros2 launch lecar_launch urdf.launch.py 
    shell_command:
    source install/setup.bash
    source ~/ros2_ws/install/setup.bash
        # - ros2 launch lecar_launch slam.launch.py
        # - ros2 launch lecar_launch localization.launch.py  
    ros2 launch lecar_launch vio.launch.py
    ```

- Terminal 3: 

    ```bash
    /home/f15car/f1tenth
    ros2 launch art_f1tenth_interface art_f1tenth_interface.launch.py
    ```


## Touble shooting

1. need to start joynode before the vesc on car
2. vio plug and unplug
3. Restart everything



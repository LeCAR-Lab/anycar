This package contains necessary modules for car dynamics modeling and model-based controller

- `model_jax`: 

    - `dbm.py`: Dynamic Bicycle Model (DBM) for vehicle dynamics modeling
    - `nn_dynamics.py`: NN Wrapper for dynamics modeling

- `controllers_jax`:
    - `mppi.py`: MPPI Implementation
    - `mppi_helper.py`: Helper functions for dealing with rollout functions

- `controllers_torch`:
    - `alt_pure_pursuit.py`: Pure Pursuit Implementation
    - `pid.py`: PID Controller Implementation
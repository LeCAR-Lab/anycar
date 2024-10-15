
# AnyCar to Anywhere: Learning Universal Dynamics Model for Agile and Adaptive Mobility
<div align="center">

[[Website]](https://lecar-lab.github.io/anycar/)
[[Arxiv]](https://arxiv.org/abs/2409.15783)
[[Video]](https://www.youtube.com/)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src="./media/2024_AnyCar.gif" width="600px"/>

</div>


## Environment Setup

> [!IMPORTANT]
> We recommend Ubuntu >= 22.04 + Python >= 3.10 + CUDA >= 12.3.
> You can create a mamba (or conda) environment before proceeding.

1. Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) on your machine
2. Clone this repo and cd to the folder

    ```bash
    git clone git@github.com:LeCAR-Lab/anycar.git
    cd anycar
3. Create a new mamba environment and activate it

    ```bash
    mamba create -n anycar python=3.10
    mamba activate anycar
    ```

4. Install python dependencies

    ```bash
    mamba install -r requirements.txt
    ```

5. Add the path of this project folder to your environment path `CAR_PATH`

6. Build the project
    ```bash
    colcon build
    ```

7. Source the environment

    ```bash
    source install/setup.bash 
    ```

8. Download [Foxglove Studio](https://foxglove.dev/download) and import the visualization config from `misc/anycar-vis.json`


## Quick Start

1. run foxglove node:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

2. run car_sim node:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py
```

3. Check the visualization in Foxglove Studio `localhost:8765`

Expected to see:

<div align="center">

<img src="./media/foxglove.gif" width="600px"/>

</div>



## AnyCar Pipeline

- Data Collection

    Please refer to [car_collect/README.md](./car_collect/README.md) for data collection.
Run scripts in `car_collect` folder to collect data, the data will be automatically saved to `car_foundation/car_foundation/data`

- Model Training

    Please refer to [car_foundation/README.md](./car_foundation/README.md) for model training.
Run scripts in `car_foundation` folder to train model, the model will be automatically saved to `car_foundation/car_foundation/models`

- Controller. 
    
    Please refer to [car_dynamics/README.md](./car_dynamics/README.md) for first-principle based dynamics model and sampling-based MPC implementation.

- Deployment. 

    The deployment pipeline (sim/real) is implemented using ROS2, please refer to [car_ros2/README.md](./car_ros2/README.md) for more details.

- Hardware Setup

    Please refer to [hardware/README.md](./hardware/README.md) for our car configurations and 3d models.


## Citation
```bibtex
@misc{xiao2024anycaranywherelearninguniversal,
      title={AnyCar to Anywhere: Learning Universal Dynamics Model for Agile and Adaptive Mobility}, 
      author={Wenli Xiao and Haoru Xue and Tony Tao and Dvij Kalaria and John M. Dolan and Guanya Shi},
      year={2024},
      eprint={2409.15783},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.15783}, 
}
```
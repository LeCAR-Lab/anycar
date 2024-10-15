from setuptools import setup

setup(
    name='car_dynamics',
    version='0.0.0',
    author='Wenli Xiao',
    author_email='wxiao2@andrew.cmu.edu',
    description='Tools for RC Car Dynamics',
    packages=['car_dynamics'],
    install_requires=[
        'flax',
        'termcolor',
        'rich',
        'scipy',
        'pandas',
        'gym',
        'pynput',
        'ipdb',
        'mujoco',
		# 'cvxpy',
		# 'casadi',
		# 'botorch==0.1.4',
		# 'gpytorch==0.3.6',
		# 'matplotlib==3.1.2',
		# 'scikit-learn==0.22.2.post1',
		# 'tikzplotlib',
    ],
)

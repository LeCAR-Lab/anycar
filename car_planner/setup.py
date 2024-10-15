from setuptools import setup

setup(
    name='car_planner',
    version='0.0.1',
    author='Haoru Xue',
    author_email='haorux@andrew.cmu.edu',
    description='car planner',
    packages=['car_planner'],
    install_requires=[
		'numpy',
		'casadi',
        'scipy',
        'jax-cosmo',
    ],
)

from setuptools import setup

setup(
    name='car_foundation',
    version='0.0.1',
    author='Haoru Xue',
    author_email='haorux@andrew.cmu.edu',
    description='car foundation',
    packages=['car_foundation'],
    install_requires=[
        'torch',
		'numpy==1.26',
		'casadi',
        'scipy',
        'flax',
        'tqdm',
        'pykan',
        'transforms3d',
        'scikit-learn',
        'wandb',
        'transformers',
        'pytorch-warmup',
        'orbax-checkpoint==0.5.15',
    ],
)

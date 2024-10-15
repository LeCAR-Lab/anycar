from setuptools import setup

setup(
   name='car_collect',
   version='0.0',
   description='A useful module',
   author='Wenli Xiao',
   author_email='',
   packages=['mujoco_collect', 'numeric_collect', 'assetto_corsa_collect', 'isaacsim_collect'],  #same as name
   install_requires=[
      'ray',
      'matplotlib',
      'jupyterlab',
      # 'torch',
      'pyproj',
      'ipywidgets',
      # 'gym',
   ], #external packages as dependencies
)
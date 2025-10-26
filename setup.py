from setuptools import setup, find_packages

setup(name='gym_abalone',
      version='0.0.1',
      packages=find_packages(exclude=['script', 'script.*']),
      install_requires=[
          'gym==0.26.2',
          'numpy<2.0.0',
          'pyglet>=2.0.0'
        ]
)
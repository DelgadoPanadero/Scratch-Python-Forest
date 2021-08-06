from setuptools import setup
from setuptools import find_packages

setup(name='skratch-python-forest',
      version='0.0',
      description='Have fun!',
      url='https://github.com/DelgadoPanadero/Scratch-Python-Forest.git',
      author='Angel Delgado',
      author_email='delgadopanadero@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={"": ["*.csv"]},
      install_requires=[
          "numpy==1.18.1",
          "matplotlib==3.4.2",
      ],
)

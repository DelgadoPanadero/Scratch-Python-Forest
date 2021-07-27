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
      install_requires=[
          "joblib==0.14.1",
          "numpy==1.18.1",
          "scipy==1.4.1",
          "sklearn==0.0",
          "scikit-learn==0.22.1"
      ],
)

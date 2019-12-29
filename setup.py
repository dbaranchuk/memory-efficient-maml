from setuptools import setup

setup(
   name='torch_maml',
   version='1.0',
   description='Gradient checkpoint technique for Meta-Agnostic-Meta-Learning',
   author='Dmitry Baranchuk',
   author_email='dmitry.baranchuk@graphics.cs.msu.ru',
   packages=['torch_maml'],
   install_requires=['torch', 'numpy']
)
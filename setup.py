from setuptools import setup

setup(
   name='torch_maml',
   version='1.0',
   description='Gradient checkpointing technique for Model Agnostic Meta Learning',
   long_description='PyTorch implementation of Model Agnostic Meta Learning with gradient checkpointing. Allows you to perform way (~10-100x) more MAML steps with the same GPU memory budget.',
   author='Dmitry Baranchuk',
   author_email='dmitry.baranchuk@graphics.cs.msu.ru',
   packages=['torch_maml'],
   license='MIT',
   install_requires=['torch>=1.1.0'],
   classifiers=[
        # Indicate who your project is intended for
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.5 ',
        'Programming Language :: Python :: 3.6 ',
        'Programming Language :: Python :: 3.7 ',

    ],

    # What does your project relate to?
    keywords='meta-learning, maml, pytorch, torch, deep learning, machine learning, gradient checkpointing, gpu',
)

# Memory efficient MAML

### Overview

PyTorch implementation of Model Agnostic Meta Learning[1] with 
 gradient checkpointing[2]. Allows you to perform way (>100x) more
 optimizer steps with the same GPU memory budget. 


### Install

For normal installation, run
```pip install torch-maml```

For development installation, clone a repo and
```python setup.py develop```


### How to use:
See examples is in [```gradient-checkpointing-maml.ipynb```](./gradient-checkpointing-maml.ipynb)

TODO colab badge


### Tips and tricks
1) Make sure that your model doesn't have implicit parameter updates like 
torch.nn.BatchNorm2d under track_running_stats=True. With gradient checkpointing
 these updates will be performed twice (once per forward pass). If still want these
 updates, take a look at [```torch-maml.utils.disable_batchnorm_stats```](./torch-maml/utils.py#L86-L101) TODOurl. 
 Note that we already support this for vanilla BatchNorm{1-3}d.

2) CUDNN optimization slows down the use of gradient checkpoints. 
One might want to set [```torch.backends.cudnn.benchmarks = False```](https://pytorch.org/docs/stable/notes/randomness.html#cudnn). 
For example, it speeds up 100 iterations of MAML on ResNet18 by 2.5x

3) When computing gradients through many MAML steps (e.g. 100 or 1000),
you really should care about vanishing and exploding gradients within
optimizers (same as in RNN). This implementation supports gradient clipping 
to avoid the explosive part of the problem.

4) Also when you deal with a large number MAML steps, be aware of 
accumulating computational error due to float precision and specifically
CUDNN operations. We recommend you to use 
```torch.backend.cudnn.determistic=True```. The problem appears when
gradients become slightly noisy due to errors and, 
during backpropagation though MAML steps, the error is likely to 
dramatically increase.  
 
### References

[1] Model Agnostic Meta Learning paper:
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

[2] Gradient checkpointing technique:
https://github.com/cybertronai/gradient-checkpointing

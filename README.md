# Gradient Checkpointing Meta-Agnostic-Meta-Learning

### Overview

Straightforward application of the gradient checkpointing 
technique to Meta-Agnostic-Meta-Learning[1], which mitigates 
its high memory consumption and allows to perform a large number
of optimizer steps in MAML. 


### Implementation

We provide PyTorch implementation of simple trick, which can be 
readily incorporated in your projects. Moreover, we wrap it as
as a pip module: ```pip install torch-maml```

Demo is in ```gradient-checkpointing-maml.ipynb```


### Dependencies

* PyTorch >= 1.1.0
* Numpy 

```pip install -r requirements.txt```

### Important notes

1) CUDNN optimization slows down the use of gradient checkpoints. 
One might want to use ```torch.backends.cudnn.benchmarks = False```. 
For example, it speeds up 100 iterations of MAML on ResNet18 by 2.5x

2) While this trick allows to perform many MAML steps (e.g. 100 or 1000),
you really should care about vanishing and exploding gradients within
optimizers like in RNN. In our demo we use gradients clipping. TODO cite 

3) Also when you deal with a large number MAML steps, be aware of 
accumulating computational errors occured on each steps due to 
CUDNN and other GPU details???, e.g. at least one can use 
```torch.backend.cudnn.determistic=True```. The problem appears when
gradients become slightly noisy due to errors and, 
during backpropagation though MAML steps, the error is likely to 
dramatically increase.  
 
### References

[1] Gradient checkpointing technique:
https://github.com/cybertronai/gradient-checkpointing

[2] Meta-Agnostic-Meta-Learning paper:
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
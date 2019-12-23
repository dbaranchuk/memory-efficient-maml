import contextlib
import gc
import time
import torch
import numpy as np
from torch import nn as nn


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def straight_through_grad(function, **kwargs):
    """
    modify function so that it is applied normally but excluded from backward pass
    :param function: callable(*inputs) -> *outputs, number and shape of outputs must match that of inputs,
    :param kwargs: keyword arguments that will be sent to each function call
    """
    def f_straight_through(*inputs):
        outputs = function(*inputs, **kwargs)
        single_output = isinstance(outputs, torch.Tensor)
        if single_output:
            outputs = [outputs]

        assert isinstance(outputs, (list, tuple)) and len(outputs) == len(inputs)
        outputs = type(outputs)(
            input + (output - input).detach()
            for input, output in zip(inputs, outputs)
        )
        return outputs[0] if single_output else outputs

    return f_straight_through


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


class OptimizerList(torch.optim.Optimizer):
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def step(self):
        return [opt.step() for opt in self.optimizers]

    def zero_grad(self):
        return [opt.zero_grad() for opt in self.optimizers]

    def add_param_group(self, *args, **kwargs):
        raise ValueError("Please call add_param_group in one of self.optimizers")

    def __getstate__(self):
        return [opt.__getstate__() for opt in self.optimizers]

    def __setstate__(self, state):
        return [opt.__setstate__(opt_state) for opt, opt_state in zip(self.optimizers, state)]

    def __repr__(self):
        return repr(self.optimizers)

    def state_dict(self, **kwargs):
        return {"opt_{}".format(i): opt.state_dict(**kwargs) for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict, **kwargs):
        return [
            opt.load_state_dict(state_dict["opt_{}".format(i)])
            for i, opt in enumerate(self.optimizers)
        ]


def get_checkpoint_steps(total_steps, checkpoint_steps=None):
    if checkpoint_steps is None:
        checkpoint_steps = max(1, int(np.ceil(np.sqrt(total_steps))))
    for _ in range(total_steps // checkpoint_steps):
        yield checkpoint_steps

    if total_steps % checkpoint_steps > 0:
        yield total_steps % checkpoint_steps


@contextlib.contextmanager
def handle_batchnorm(model:nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            assert module.train
    try:
        yield
    finally:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                assert module.train
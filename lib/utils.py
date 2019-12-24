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


def reset_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.batchnorm._BatchNorm):
            module.reset_running_stats()


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
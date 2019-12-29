import contextlib
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

DUMMY_TENSOR = torch.tensor([])

DEFAULT_MEMO = dict()


def copy_and_replace(original, replace=None, do_not_copy=None):
    """
    A convenience function for creating modified copies out-of-place with deepcopy
    :param original: object to be copied
    :param replace: a dictionary {old object -> new object}, replace all occurences of old object with new object
    :param do_not_copy: a sequence of objects that will not be copied (but may be replaced)
    :return: a copy of obj with replacements
    """
    replace, do_not_copy = replace or {}, do_not_copy or {}
    memo = dict(DEFAULT_MEMO)
    for item in do_not_copy:
        memo[id(item)] = item

    for item, replacement in replace.items():
        memo[id(item)] = replacement

    return deepcopy(original, memo)


@contextmanager
def do_not_copy(*items):
    """ all calls to copy_and_replace within this context won't copy items (but can replace them) """
    global DEFAULT_MEMO
    keys_to_remove = []
    for item in items:
        key = id(item)
        if key in DEFAULT_MEMO:
            DEFAULT_MEMO[key] = item
            keys_to_remove.append(key)

    yield

    for key in keys_to_remove:
        DEFAULT_MEMO.pop(key)


def straight_through_grad(function, **kwargs):
    """
    Modify function so that it is applied normally but excluded from backward pass
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


class ClipGradNorm(torch.autograd.Function):
    """ Do nothing on forward pass but clip gradients by their global norm during backward pass """
    @staticmethod
    def forward(ctx, *inputs):
        """ We manually put max_norm to the end of tensor inputs in order to please Function.apply """
        ctx._max_norm = inputs[-1].item()
        return inputs[:-1]

    @staticmethod
    def backward(ctx, *grad_outputs):
        global_grad_norm = sum([grad_output.norm() ** 2 for grad_output in grad_outputs]).sqrt().item()
        clip_grad_norm = lambda grad: grad * (ctx._max_norm / max(ctx._max_norm, global_grad_norm))
        grad_inputs = tuple(map(clip_grad_norm, grad_outputs))
        return grad_inputs + (None, )


@contextlib.contextmanager
def disable_batchnorm_stats(model: nn.Module):
    """
    Turns off batchnorm stats tracking inside the context
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            assert module.train
    try:
        yield
    finally:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True
                assert module.train

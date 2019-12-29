import contextlib
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

NONE_TENSOR = torch.tensor([])


def is_none_tensor(x):
    return isinstance(x, torch.Tensor) and tuple(x.shape) == (0,)


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


# ----------------------------------------------------------------------------------- #
#                                Nested structures                                    #
# utility functions that help you process nested dicts, tuples, lists and namedtuples #


def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True

    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True

    else:
        return True


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[
            _nested_pack(flat_iter, x)
            for x in structure]
            )
    if isinstance(structure, (list, tuple)):
        return type(structure)(
            _nested_pack(flat_iter, x)
            for x in structure
            )
    elif isinstance(structure, dict):
        return {
            k: _nested_pack(flat_iter, v)
            for k, v in sorted(structure.items())
            }
    else:
        return next(flat_iter)


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def nested_map(fn, *t):
    # Check arguments.
    if not t:
        raise ValueError('Expected 2+ arguments, got 1')
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = 'Nested structure of %r and %r differs'
            raise ValueError(msg % (t[0], t[i]))

    # Map.
    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])
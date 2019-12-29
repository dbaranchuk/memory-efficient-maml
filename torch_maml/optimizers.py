"""
Utilities required for backpropagating through gradient descent steps
"""
from collections import namedtuple
from warnings import warn
from itertools import chain

import torch
from torch import nn as nn
from .utils import straight_through_grad, copy_and_replace, ClipGradNorm, NONE_TENSOR, is_none_tensor


def get_updated_model(model: nn.Module, loss=None, gradients=None, parameters=None,
                      detach=False, learning_rate=1.0, allow_unused=False,
                      max_grad_grad_norm=None, **kwargs):
    """
    Creates a copy of model whose parameters are updated with one-step gradient descent w.r.t. loss
    The copy will propagate gradients into the original model
    :param model: original model
    :param loss: scalar objective to backprop from; provide either this or gradients
    :param gradients: a list or tuple of gradients (updates) for each parameter; provide either this or loss
    :param parameters: list/tuple of parameters to update, defaults to model.parameters()
    :param detach: if True, the resulting model will not propagate gradients to the original model
    :param learning_rate: scales gradients by this value before updating
    :param allow_unused: by default, raise an error if one or more parameters receive None gradients
        Otherwise (allow_unused=True) simply do not update these parameters
    :param max_grad_grad_norm: maximal global norm of gradients passing through optimization procedure.
        Used for gradient clipping through optimizer steps.
    """
    assert (loss is None) != (gradients is None)
    parameters = list(model.parameters() if parameters is None else parameters)
    if gradients is None:
        assert torch.is_grad_enabled()
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=not detach, only_inputs=True, allow_unused=allow_unused, **kwargs)

    assert isinstance(gradients, (list, tuple)) and len(gradients) == len(parameters)

    updates = []
    for weight, grad in zip(parameters, gradients):
        if grad is not None:
            update = weight - learning_rate * grad
            if detach:
                update = update.detach().requires_grad_(weight.requires_grad)
            updates.append(update)

    if max_grad_grad_norm is not None:
        updates = ClipGradNorm.apply(*(updates + [torch.as_tensor(max_grad_grad_norm)]))
    updates = dict(zip(parameters, updates))

    do_not_copy = [tensor for tensor in chain(model.parameters(), model.buffers())
                   if tensor not in updates]

    return copy_and_replace(model, updates, do_not_copy)


class IngraphGradientDescent(nn.Module):
    """ Optimizer that updates model out-of-place and returns a copy with changed parameters """
    OptimizerState = namedtuple("OptimizerState", [])

    def __init__(self, learning_rate=1.0):
        super().__init__()
        self.learning_rate = learning_rate

    def get_initial_state(self, module, *, parameters: list, **kwargs):
        """ 
        Return initial optimizer state: momenta, rms, etc. 
        State:
        * must be a (nested) collection of torch tensors : lists/tuples/dicts/namedtuples of tensors or lists/... of them
        * the structure (i.e. lengths) of this collection should NOT change between iterations.
        * the optimizer state at the input of :step: method forces requires_grad=True to all tensors. 
        """
        return self.OptimizerState()

    def step(self, state: OptimizerState, module: nn.Module, loss, parameters=None, **kwargs):
        """
        Return an updated copy of model after one iteration of gradient descent
        :param state: optimizer state (as in self.get_initial_state)
        :param module: module to be updated
        :param loss: torch scalar that is differentiable w.r.t. model parameters
        :parameters: parameters of :module: that will be edited by updates (default = module.parameters())
        :param kwargs: extra parameters passed to get_updated_model
        :returns: new_state, updated_self
            new_state: self.OptimizerState - optimizer state after performing sgd step
            updated_self: updated(out-of-place) version of self
        """
        updated_model = get_updated_model(module, loss=loss, learning_rate=self.learning_rate,
                                          parameters=list(parameters or module.parameters()), **kwargs)
        return state, updated_model

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class IngraphRMSProp(IngraphGradientDescent):
    OptimizerState = namedtuple(
        "OptimizerState", ["grad_momenta", "ewma_grad_norms_sq", "learning_rate", "momentum", "beta", "epsilon"])

    def __init__(self, learning_rate=None, log_learning_rate=None, momentum=None, beta=None,
                 epsilon=None, log_epsilon=None, force_trainable_params=False):
        """
        Ingraph optimzier that performs RMSProp updates with optional momentum
        :param learning_rate: log(alpha) for gradient descent, all updates are scaled by exponent of this value
        :param momentum: momentum coefficient, the update direction is (1 - momentum) * prev_update  + update,
            default = no momentum
        :param beta: RMSProp decay coefficient, the update is scaled by 1 / sqrt(ewma + epsilon)
            where ewma = prev_ewma * beta + dL/dw ^ 2 * (1 - beta), default = no RMSProp
        :param force_trainable_params: if True, treats all optimizer parameters that are not None as learnable
            parameters that are trained alongside other non-edited layers

        """
        nn.Module.__init__(self)
        self.hparams = dict(
            learning_rate=learning_rate, log_learning_rate=log_learning_rate,
            momentum=momentum, beta=beta, epsilon=epsilon, log_epsilon=log_epsilon
        )

        if force_trainable_params:
            for key in self.hparams:
                if self.hparams[key] is None: continue
                elif isinstance(self.hparams[key], nn.Parameter): continue
                elif isinstance(self.hparams[key], torch.Tensor) and self.hparams[key].requires_grad: continue
                self.hparams[key] = nn.Parameter(torch.as_tensor(self.hparams[key]))

        for key in self.hparams:
            if isinstance(self.hparams[key], nn.Parameter):
                self.register_parameter(key, self.hparams[key])

    def get_initial_state(self, module: nn.Module, *, parameters: list, **overrides):
        """
        Create initial state and make sure all parameters are in a valid range.
        State:
        * must be a (nested) collection of torch tensors : lists/tuples/dicts/namedtuples of tensors or lists/... of them
        * the structure (i.e. lengths) of this collection should NOT change between iterations.
        * the optimizer state at the input of :step: method forces requires_grad=True to all tensors. 
        
        :param module: module to be updated
        :param parameters: list of trainable parameters
        :param overrides: send key-value optimizer params with same names as at init to override them
        :return: self.OptimizerState
        """
        for key in overrides:
            assert key in self.hparams, "unknown optimizer parameter {}".format(key)
        hparams = dict(self.hparams, **overrides)

        assert (hparams['learning_rate'] is None) != (hparams['log_learning_rate'] is None), "provide lr or log lr"
        learning_rate = hparams['learning_rate'] or torch.exp(hparams['log_learning_rate'])
        learning_rate = straight_through_grad(torch.clamp_min, min=0.0)(torch.as_tensor(learning_rate))

        momentum = hparams.get('momentum')
        if momentum is not None:
            momentum = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(momentum))
        else:
            momentum = NONE_TENSOR

        if isinstance(momentum, torch.Tensor) and momentum.requires_grad:
            warn("The derivative of updated params w.r.t. momentum is proportional to momentum^{n_steps - 1}, "
                 "optimizing it with gradient descent may suffer from poor numerical stability.")

        beta = hparams.get('beta')
        if beta is not None:
            beta = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(beta))

            assert hparams['epsilon'] is None or hparams['log_epsilon'] is None, "provide either epsilon or log epsilon"
            if hparams['epsilon'] is None and hparams['log_epsilon'] is None:
                hparams['epsilon'] = 1e-6
            epsilon = torch.as_tensor(hparams['epsilon']) or torch.exp(hparams['log_epsilon'])
            epsilon = straight_through_grad(torch.clamp_min, min=1e-9)(torch.as_tensor(epsilon))

        else:
            epsilon = NONE_TENSOR
        
        dummy_grad_momenta = dummy_ewma = [NONE_TENSOR for _ in parameters]
        return self.OptimizerState(dummy_grad_momenta, dummy_ewma, learning_rate, momentum, beta, epsilon)

    def step(self, state: OptimizerState, module: nn.Module, loss, parameters=None, **kwargs):
        """
        :param state: optimizer state (as in self.get_initial_state)
        :param module: module to be edited
        :param loss: torch scalar that is differentiable w.r.t. model parameters
        :param parameters: if model
        :param kwargs: extra parameters passed to get_updated_model
        :returns: new_state, updated_self
            new_state: self.OptimizerState - optimizer state after performing sgd step
            updated_self: updated copy of module
        """
        grad_momenta, ewma_grad_norms_sq, learning_rate, momentum, beta, epsilon = state
        learning_rate, momentum, beta, epsilon = [
            tensor.to(loss.device) for tensor in (learning_rate, momentum, beta, epsilon)]
        parameters = list(parameters or module.parameters())
        gradients = torch.autograd.grad(loss, parameters, create_graph=True, only_inputs=True, allow_unused=False)
        updates = list(gradients)  # updates are the scaled/accumulated/tuned gradients
        
        if not is_none_tensor(momentum) and float(momentum) != 0:
            # momentum: accumulate gradients with moving average-like procedure
            if is_none_tensor(grad_momenta[0]):
                grad_momenta = list(gradients)
            else:
                for i in range(len(grad_momenta)):
                    grad_momenta[i] = grad_momenta[i] * momentum + gradients[i]
            updates = grad_momenta

        if not is_none_tensor(beta) and float(beta) != 0:
            # RMSProp: first, update the moving average squared norms
            if is_none_tensor(ewma_grad_norms_sq[0]):
                ewma_grad_norms_sq = list(map(lambda g: g ** 2, gradients))
            else:
                for i in range(len(ewma_grad_norms_sq)):
                    ewma_grad_norms_sq[i] = beta * ewma_grad_norms_sq[i] + (1.0 - beta) * gradients[i] ** 2

            # scale updates by 1 / sqrt(moving_average_norm_squared + epsilon)
            for i in range(len(updates)):
                updates[i] = updates[i] / torch.sqrt(ewma_grad_norms_sq[i] + epsilon)

        # finally, perform sgd update
        updated_module = get_updated_model(module, loss=None, gradients=updates, parameters=parameters,
                                           learning_rate=learning_rate, **kwargs)
        new_state = self.OptimizerState(grad_momenta, ewma_grad_norms_sq, learning_rate, momentum, beta, epsilon)
        return new_state, updated_module

    def extra_repr(self):
        return repr(self.hparams)

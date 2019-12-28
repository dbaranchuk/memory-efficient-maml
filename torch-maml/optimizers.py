"""
Utilities required for backpropagating through gradient descent steps
"""
from collections import namedtuple
from warnings import warn
from itertools import chain

import torch
from torch import nn as nn
from .utils import straight_through_grad, copy_and_replace


def get_updated_model(model: nn.Module, loss=None, gradients=None, parameters=None,
                      detach=False, learning_rate=1.0, allow_unused=False,
                      norm_grad=True, max_grad_norm=1e4, **kwargs):
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
    :param norm_grad: TODO
    :param max_grad_norm: TODO
    """
    assert (loss is None) != (gradients is None)
    parameters = list(model.parameters() if parameters is None else parameters)
    if gradients is None:
        assert torch.is_grad_enabled()
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=not detach, only_inputs=True, allow_unused=allow_unused, **kwargs)

    assert isinstance(gradients, (list, tuple)) and len(gradients) == len(parameters)

    # Hook to normalize weight gradients after each optimizer step
    def normalize_grad(grad):
        return (grad * max_grad_norm) / max(grad.norm(), max_grad_norm)

    updates = dict()
    for weight, grad in zip(parameters, gradients):
        if grad is not None:
            update = weight - learning_rate * grad
            if norm_grad:
                # TODO: check whether the last iteration is taken into account
                update.register_hook(normalize_grad)
            if detach:
                update = update.detach().requires_grad_(weight.requires_grad)
            updates[weight] = update

    do_not_copy = [tensor for tensor in chain(model.parameters(), model.buffers())
                   if tensor not in updates]

    return copy_and_replace(model, updates, do_not_copy)


class IngraphGradientDescent(nn.Module):
    """ Optimizer that updates model out-of-place and returns a copy with changed parameters """
    OptimizerState = namedtuple("OptimizerState", [])

    def __init__(self, learning_rate=1.0):
        super().__init__()
        self.learning_rate = learning_rate

    def get_initial_state(self, editable, **kwargs):
        """ Return initial optimizer state: momenta, rms, etc. """
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
        self.params = dict(
            learning_rate=learning_rate, max_grad_norm=max_grad_norm, learning_rate=log_learning_rate,
            momentum=momentum, beta=beta, epsilon=epsilon, log_epsilon=log_epsilon
        )

        if force_trainable_params:
            for key in self.params:
                if self.params[key] is None: continue
                elif isinstance(self.params[key], nn.Parameter): continue
                elif isinstance(self.params[key], torch.Tensor) and self.params[key].requires_grad: continue
                self.params[key] = nn.Parameter(torch.as_tensor(self.params[key]))

        for key in self.params:
            if isinstance(self.params[key], nn.Parameter):
                self.register_parameter(key, self.params[key])

    def get_initial_state(self, editable, **overrides):
        """
        Create initial state and make sure all parameters are in a valid range
        :param editable: module to be edited
        :param overrides: send key-value optimizer params with same names as at init to override them
        :return: Editable.OptimizerState
        """
        for key in overrides:
            assert key in self.params, "unknown optimizer parameter {}".format(key)
        params = dict(self.params, **overrides)

        assert (params['learning_rate'] is None) != (params['log_learning_rate'] is None), "provide either lr or log lr"
        learning_rate = params['learning_rate'] or torch.exp(params['log_learning_rate'])
        learning_rate = straight_through_grad(torch.clamp_min, min=0.0)(torch.as_tensor(learning_rate))

        momentum = params.get('momentum')
        if momentum is not None:
            momentum = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(momentum))
        if isinstance(momentum, torch.Tensor) and momentum.requires_grad:
            warn("The derivative of updated params w.r.t. momentum is proportional to momentum^{n_steps - 1}, "
                 "optimizing it with gradient descent may suffer from poor numerical stability.")

        beta = params.get('beta')
        if beta is not None:
            beta = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(beta))

            assert params['epsilon'] is None or params['log_epsilon'] is None, "provide either epsilon or log epsilon"
            if params['epsilon'] is None and params['log_epsilon'] is None:
                params['epsilon'] = 1e-6
            epsilon = params['epsilon'] or torch.exp(params['log_epsilon'])
            epsilon = straight_through_grad(torch.clamp_min, min=1e-9)(torch.as_tensor(epsilon))

        else:
            epsilon = None

        return self.OptimizerState(None, None, learning_rate, momentum, beta, epsilon)

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
        parameters = list(parameters or module.parameters())
        gradients = torch.autograd.grad(loss, parameters, create_graph=True, only_inputs=True, allow_unused=False)
        updates = list(gradients)  # updates are the scaled/accumulated/tuned gradients

        if momentum is not None:
            # momentum: accumulate gradients with moving average-like procedure
            if grad_momenta is None:
                grad_momenta = list(gradients)
            else:
                for i in range(len(grad_momenta)):
                    grad_momenta[i] = grad_momenta[i] * momentum + gradients[i]
            updates = grad_momenta

        if beta is not None:
            # RMSProp: first, update the moving average squared norms
            if ewma_grad_norms_sq is None:
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
        return repr(self.params)

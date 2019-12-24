from collections import namedtuple
from itertools import chain
import torch
import torch.nn as nn
from copy import deepcopy

from lib.ingraph_update import IngraphGradientDescent
from lib.copy_and_replace import copy_and_replace, do_not_copy
from lib.utils import handle_batchnorm, reset_batchnorm
from torch.utils.checkpoint import checkpoint


class GradientCheckpointMAML:
    Result = namedtuple('Result', ['model', 'loss', 'optimizer_state'])

    def __init__(self, model:nn.Module, loss_function,
                 checkpoint_steps=1,
                 optimizer=IngraphGradientDescent(0.01),
                 get_parameters=nn.Module.parameters):
        """ MAML module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be updated
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param meta_optimizer: in-graph optimizer that creates updated copies of model
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.get_parameters = get_parameters
        self.checkpoint_steps = checkpoint_steps

    def __call__(self, inputs, opt_kwargs=None, loss_kwargs=None, **kwargs):
        """
        Perform meta optimizer to the model (out-of-place) and return an updated copy
        :param inputs: data that is fed into the model
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updated_model, final_loss
        :rtype: MAML.Result
        """
        model = self.model
        opt_kwargs, loss_kwargs = opt_kwargs or {}, loss_kwargs or {}
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)

        # Reset stats for nn.BatchNorm2d TODO
        # reset_batchnorm(model)
        print("MODEL UNIQUE ID", id(model))

        parameters_to_copy = list(self.get_parameters(model))
        parameters_not_to_copy = [param for param in chain(model.parameters(), model.buffers())
                                  if param not in set(parameters_to_copy)]

        # WARNING: this code treats model, parameters_to_copy, parameters_not_to_copy]
        # as a global variables for _maml_internal. Please DO NOT change or delete them in this function
        def _maml_internal(i, steps, *trainable_parameters_and_state):
            trainable_parameters = trainable_parameters_and_state[:len(parameters_to_copy)]
            optimizer_state = trainable_parameters_and_state[len(parameters_to_copy):]

            updated_model = copy_and_replace(
                model, dict(zip(parameters_to_copy, trainable_parameters)), parameters_not_to_copy)
            # version of model with specified initial parameters

            inside_checkpoint_forward = not torch.is_grad_enabled()

            for _ in range(steps.item()):
                print("MAML INTERNAL STEP:", int(i),
                      "inside_checkpoint_forward:", inside_checkpoint_forward)
                with torch.enable_grad():
                    with handle_batchnorm(updated_model):
                        index = int(i.item())
                        loss = self.loss_function(updated_model, inputs[index], **loss_kwargs)

                    with do_not_copy(*parameters_not_to_copy):
                        optimizer_state, updated_model = self.optimizer.step(optimizer_state,
                            updated_model, loss=loss, detach=inside_checkpoint_forward,
                            parameters=self.get_parameters(updated_model), **kwargs)
                i = i + 1
            return (i, loss, *self.get_parameters(updated_model), *optimizer_state)

        i = torch.zeros(1, requires_grad=True)
        trainable_parameters = self.get_parameters(model)
        trainable_parameters_and_optimizer_state = list(chain(trainable_parameters, optimizer_state))

        for chunk_start in range(0, len(inputs), self.checkpoint_steps):
            steps = min(self.checkpoint_steps, len(inputs) - chunk_start)
            i, loss, *trainable_parameters_and_optimizer_state = checkpoint(
                _maml_internal, i, torch.as_tensor(steps), *trainable_parameters_and_optimizer_state)

        trainable_parameters = trainable_parameters_and_optimizer_state[:len(parameters_to_copy)]
        optimizer_state = trainable_parameters_and_optimizer_state[len(parameters_to_copy):]

        updated_model = copy_and_replace(
            model, dict(zip(self.get_parameters(model), trainable_parameters)),
            parameters_not_to_copy)
        return self.Result(updated_model, loss=loss, optimizer_state=optimizer_state)


class MAML:
    Result = namedtuple('Result', ['model', 'loss', 'optimizer_state'])

    def __init__(self, model: nn.Module, loss_function,
                 optimizer=IngraphGradientDescent(0.01),
                 get_parameters=nn.Module.parameters):
        """ MAML module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be updated
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param optimizer: in-graph optimizer that creates updated copies of model
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.get_parameters = get_parameters

    def __call__(self, inputs, opt_kwargs=None, loss_kwargs=None, **kwargs):
        """
        Perform meta optimizer to the model (out-of-place) and return an updated copy
        :param inputs: data that is fed into the model
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updated_model, final_loss
        :rtype: MAML.Result
        """
        opt_kwargs, loss_kwargs = opt_kwargs or {}, loss_kwargs or {}
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)

        updated_model = self.model
        # Reset stats for nn.BatchNorm2d
        # assert not self.model.training
        #reset_batchnorm(updated_model) #TODO this loses accumulated statistics #TODO all 3 batchnorm types

        for input in inputs:
            loss = self.loss_function(updated_model, input, **loss_kwargs)
            optimizer_state, updated_model = self.optimizer.step(optimizer_state, updated_model, loss,
                                                              parameters=self.get_parameters(updated_model), **kwargs)

        return self.Result(updated_model, loss=loss, optimizer_state=optimizer_state)
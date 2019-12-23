from collections import namedtuple
from itertools import chain
import torch
import torch.nn as nn

from lib.ingraph_update import IngraphGradientDescent
from lib.copy_and_replace import copy_and_replace, do_not_copy
from lib.utils import get_checkpoint_steps, handle_batchnorm, reset_batchnorm
from torch.utils.checkpoint import checkpoint


class GradientCheckpointMAML:
    Result = namedtuple('Result', ['model', 'loss'])

    def __init__(self, model:nn.Module, loss_function,
                 meta_optimizer=IngraphGradientDescent(0.01),
                 get_parameters=nn.Module.parameters,
                 max_steps=50):
        """ MAML module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be updated
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param meta_optimizer: in-graph optimizer that creates updated copies of model
            :param max_steps: after this many gradient steps the process is terminated
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.meta_optimizer = meta_optimizer
        self.get_parameters = get_parameters
        self.max_steps = max_steps

    def __call__(self, inputs, checkpoint_steps=None, opt_kwargs=None, **kwargs):
        """
        Perform meta optimizer to the model (out-of-place) and return an updated copy
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updated_model, final_loss
        :rtype: MAML.Result
        """
        model = self.model
        opt_kwargs = opt_kwargs or {}
        max_steps = min(len(inputs), self.max_steps)
        optimizer_state = self.meta_optimizer.get_initial_state(self, **opt_kwargs)

        # Reset stats for nn.BatchNorm2d
        reset_batchnorm(model)
        print("MODEL UNIQUE ID", id(model))

        parameters_to_copy = list(self.get_parameters(model))
        parameters_not_to_copy = [param for param in chain(model.parameters(), model.buffers())
                                  if param not in set(parameters_to_copy)]

        # WARNING: this code treats model, parameters_to_copy, parameters_not_to_copy]
        # as a global variables for _maml_internal. Please DO NOT change or delete them in this function
        def _maml_internal(i, steps, *trainable_parameters):
            updated_model = copy_and_replace(
                model, dict(zip(parameters_to_copy, trainable_parameters)), parameters_not_to_copy)
            # version of model with specified initial parameters

            inside_checkpoint_forward = not torch.is_grad_enabled()

            for _ in range(steps.item()):
                print("MAML INTERNAL STEP:", int(i),
                      "inside_checkpoint_forward:", inside_checkpoint_forward)
                with torch.enable_grad():
                    #with handle_batchnorm(updated_model):
                    index = int(i.item())
                    loss = self.loss_function(updated_model, inputs[index], **kwargs)

                    with do_not_copy(*parameters_not_to_copy):
                        _, updated_model = self.meta_optimizer.step(optimizer_state,
                            updated_model, loss=loss, detach=inside_checkpoint_forward,
                            parameters=self.get_parameters(updated_model))
                i = i + 1
            return (i, loss, *self.get_parameters(updated_model))

        i = torch.zeros(1, requires_grad=True)
        trainable_parameters = self.get_parameters(model)

        for steps in get_checkpoint_steps(max_steps, checkpoint_steps):
            i, loss, *trainable_parameters = checkpoint(
                _maml_internal, i, torch.as_tensor(steps), *trainable_parameters)

        updated_model = copy_and_replace(
            model, dict(zip(self.get_parameters(model), trainable_parameters)), parameters_not_to_copy)
        return self.Result(updated_model, loss=loss)


    # def __call__(self, inputs, opt_kwargs=None, **kwargs):
    #     """
    #     Perform meta optimizer to the model (out-of-place) and return an updated copy
    #     :param opt_kwargs: optional overrides for optimizer.get_initial_state
    #     :param kwargs: extra parameters passed to optimizer.step
    #     :returns: updated_model, final_loss
    #     :rtype: MAML.Result
    #     """
    #     opt_kwargs = opt_kwargs or {}
    #     optimizer_state = self.meta_optimizer.get_initial_state(self, **opt_kwargs)
    #     model = self.model
    #     # Reset stats for nn.BatchNorm2d
    #     reset_batchnorm(model)
    #     for input in inputs:
    #         loss = self.loss_function(model, input)
    #         optimizer_state, model = self.meta_optimizer.step(optimizer_state, model, loss,
    #                                                      parameters=self.get_parameters(model), **kwargs)
    #     return self.Result(model, loss=loss)
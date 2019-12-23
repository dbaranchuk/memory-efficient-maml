from collections import namedtuple
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.ingraph_update import IngraphGradientDescent
from lib.copy_and_replace import copy_and_replace, do_not_copy
from lib.utils import get_checkpoint_steps, handle_batchnorm
from torch.utils.checkpoint import checkpoint


class MAML(nn.Module):
    Result = namedtuple('Result', ['model', 'loss'])

    def __init__(self, model, loss_function=F.cross_entropy,
                 optimizer=IngraphGradientDescent(0.01), max_steps=50):
        """ MAML module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be edited
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param optimizer: in-graph optimizer that creates updated copies of model
        """
        super().__init__()
        self.module, self.loss_function, self.optimizer = model, loss_function, optimizer
        self.max_steps = max_steps
        self.model = model

    @staticmethod
    def reset_batchnorm(model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

    def forward(self, x):
        return self.model(x)

    def meta_forward(self, inputs, targets, max_steps=None, opt_kwargs=None, **kwargs):
        """
        Perform meta optimizer to the model (out-of-place) and return an updated copy
        :param inputs: data that is fed into the model
        :param targets: reference answers that are fed into loss function
        :param max_steps: after this many gradient steps the process is terminated
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updaed_model, final_loss
        :rtype: MAML.Result
        """
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)
        model = self
        max_steps = max_steps or self.max_steps

        # Reset stats for nn.BatchNorm2d
        model.reset_batchnorm(model)

        for _ in range(max_steps):
            prediction = model(inputs)
            loss = self.loss_function(prediction, targets)

            optimizer_state, model = self.optimizer.step(
                optimizer_state, model, loss, parameters=model.get_trainable_parameters(model), **kwargs)
        return self.Result(model, loss=loss)

    def checkpoint_meta_forward(self, inputs, targets, total_steps, checkpoint_steps=None,
                                get_parameters=nn.Module.parameters, opt_kwargs=None, **kwargs):
        model = self
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)

        # Reset stats for nn.BatchNorm2d
        model.reset_batchnorm(model)

        print("MODEL UNIQUE ID", id(model))
        #     assert not model.training, "randomness and batchnorm-like layers not yet supported"

        parameters_to_copy = list(model.get_trainable_parameters(model))
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
                i = i + 1
                print("MAML INTERNAL STEP:", int(i),
                      "inside_checkpoint_forward:", inside_checkpoint_forward)
                with torch.enable_grad():
                    with handle_batchnorm(updated_model):
                        preds = updated_model(inputs)
                    loss = F.cross_entropy(preds, targets)
                    with do_not_copy(*parameters_not_to_copy):
                        optimizer_state, updated_model = self.optimizer.step(optimizer_state,
                            updated_model, loss=loss, detach=inside_checkpoint_forward,
                            parameters=updated_model.get_trainable_parameters(updated_model))
            return (i, loss, *get_parameters(updated_model))

        i = torch.zeros(1, requires_grad=True)
        trainable_parameters = model.get_trainable_parameters(model)

        for steps in get_checkpoint_steps(total_steps, checkpoint_steps):
            i, loss, *trainable_parameters = checkpoint(
                _maml_internal, i, torch.as_tensor(steps), *trainable_parameters)
        assert i == total_steps
        updated_model = copy_and_replace(
            model, dict(zip(get_parameters(model), trainable_parameters)), parameters_not_to_copy)
        return self.Result(updated_model, loss=loss)
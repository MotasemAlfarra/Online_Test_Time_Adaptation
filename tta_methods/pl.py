from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

# This code is adapted from LAME implementation of PL:
# https://github.com/fiveai/LAME/blob/master/src/adaptation/pseudo_label.py
# Check https://github.com/fiveai/LAME/tree/master/configs/method/default for optimal parameters for tent,shot,adabn,lame

# BEST LR: lr=0.001
class PSEUDOLABEL(nn.Module):
    """
    PL Method
    """
    def __init__(self, model, args, **kwargs):
        super().__init__()

        self.model, self.optimizer = prepare_shot_model_and_optimizer(model, args)
        self.steps = args.steps #Modify this based on optimal parameters
        self.episodic = False # Put true to reset parameters to original ones each forward
        self.threshold = args.threshold
        self.num_classes = 1000 #hardcoded for imagenet
        
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        if self.episodic:
            self.model_state, self.optimizer_state = \
                copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        outputs = model(x)

        softmax_out = outputs.softmax(1)
        max_prob = torch.max(softmax_out.detach(), dim=-1).values
        mask = max_prob > self.threshold


        if mask.sum():
            hot_labels = F.one_hot(softmax_out[mask].argmax(-1).detach(), self.num_classes)
            loss = - (hot_labels * torch.log(softmax_out[mask] + 1e-10)).sum(-1).mean()
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return outputs
    
def prepare_shot_model_and_optimizer(model, args):
    model = configure_model(model)
    params, _ = collect_params(model)
    # hardcoded best params for imagenet-c
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    check_model(model)
    return model, optimizer


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
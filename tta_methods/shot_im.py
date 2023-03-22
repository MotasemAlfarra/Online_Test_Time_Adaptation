from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
# This code is adapted from LAME implementation of shot:
# https://github.com/fiveai/LAME/blob/master/src/adaptation/shot.py
# Check https://github.com/fiveai/LAME/tree/master/configs/method/default for optimal parameters for tent,shot,adabn,lame
class SHOTIM(nn.Module):
    """
    SHOTIM method from https://arxiv.org/abs/2002.08546
    """
    def __init__(self, model, args):
        super().__init__()

        self.model, self.optimizer = self.prepare_shot_model_and_optimizer(model, args)
        self.steps = args.steps #Modify this based on optimal parameters
        self.episodic = False # Put true to reset parameters to original ones each forward

        self.beta_clustering_loss = args.beta_clustering_loss
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

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def prepare_shot_model_and_optimizer(self, model, args):
        model = self.configure_model(model, args.update_bn_only)
        params, _ = self.collect_params(model)
        # LR and Momentum Optimal for SHOT and SHOT-IM harcoded here
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        return model, optimizer


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        if self.beta_clustering_loss:
            outputs, features = model(x, return_feature=True)
        else:
            outputs = model(x)

        softmax_out = outputs.softmax(1)
        msoftmax = softmax_out.mean(0)

        # SHOT-IM
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs


    def collect_params(self, model):
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
    
    def configure_model(self, model, update_bn_only=True):
        """Configure model for use with tent."""
        # train mode, because shot optimizes the model to minimize entropy
        # SHOT updates all parameters of the feature extractor, excluding the last FC layers
        # Original SHOT implementation
        if not update_bn_only:
            model.train()
            # is this needed? review later
            model.requires_grad_(True)
            # Freeze FC layers
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.requires_grad_(False)
                    
        else:
            # In case we want shot to update only the BN layers
            # disable grad, to (re-)enable only what tent updates (originally not used by shot but other papers use it when using shot)
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

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
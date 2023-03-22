from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from tte_methods.shot_im import SHOTIM, copy_model_and_optimizer, load_model_and_optimizer
# This code is adapted from LAME implementation of shot:
# https://github.com/fiveai/LAME/blob/master/src/adaptation/shot.py
# Check https://github.com/fiveai/LAME/tree/master/configs/method/default for optimal parameters for tent,shot,adabn,lame
class SHOT(SHOTIM):
    """
    SHOTIM method from https://arxiv.org/abs/2002.08546
    """
    def __init__(self, model, args):
        super().__init__(model, args)

        self.model, self.optimizer = self.prepare_shot_model_and_optimizer(model, args)
        self.steps = args.steps #Modify this based on optimal parameters
        self.episodic = False # Put true to reset parameters to original ones each forward

        assert args.beta_clustering_loss > 0, "beta_clustering_loss must be > 0, otherwise use SHOTIM"
        self.beta_clustering_loss = args.beta_clustering_loss
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        if self.episodic:
            self.model_state, self.optimizer_state = \
                copy_model_and_optimizer(self.model, self.optimizer)
                
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

        # ================ SHOT-IM ================
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
        # ================ SHOT-IM ================
        
        
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        # normalize features
        features = features / features.norm(dim=1, keepdim=True)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L386
        # features = (features.t()/torch.norm(features,p=2,dim=1)).t()
        
        # Compute clustering loss
        # Compute centroids of each class            
        K = outputs.shape[1]
        aff = softmax_out
        
        initial_centroids = torch.matmul(aff.t(), features)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L391
        # initial_centroids = aff.transpose().dot(features)
        
        #normalize centroids
        initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
        # aff.sum(0, keepdim=True).t() is equivalente to aff.sum(0)[:, None]

        # Compute distances to centroids
        distances = torch.cdist(features, initial_centroids, p=2)
        # Compute pseudo labels
        pseudo_labels = distances.argmin(axis=1)
        
        # I don't know why they do this, but it's in the original implementation
        for _ in range(1):
            aff = torch.eye(K)[pseudo_labels].to(aff.device)
            initial_centroids = torch.matmul(aff.t(), features)
            initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
            distances = torch.cdist(features, initial_centroids, p=2)
            pseudo_labels = distances.argmin(axis=1)
            
        # Compute clustering loss
        loss += self.beta_clustering_loss * F.cross_entropy(outputs, pseudo_labels)
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs
import torch.nn as nn
import torch
import torch.optim as optim
from copy import deepcopy
from utils.memo_utils.third_party_memo import aug
import numpy as np
from utils.memo_utils.memo_transforms import te_transforms_inc
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# Batch size for memo is 64 (number of augmentations)
class Memo(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.batch_size = 64
        self.model = model
        self.model_state_dict, self.model_per_image, self.optimizer = copy_model_and_optimizer(self.model)
        self.niter = 1 #args.niter
        self.prior_strength =  16 #args.prior_strength
        self.criterion = marginal_entropy
     

    def forward(self, x):
        self.reset()
        self.forward_and_adapt_memo(x, self.criterion)
        outputs = self.test_single(x)
        return outputs

    def test_single(self, image):
        self.model_per_image.eval()

        if self.prior_strength < 0:
            nn.BatchNorm2d.prior = 1
        else:
            nn.BatchNorm2d.prior = float(self.prior_strength) / float(self.prior_strength + 1)
        transform = te_transforms_inc 
        inputs = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model_per_image(inputs.cuda())
            # _, predicted = outputs.max(1)
            # confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
        nn.BatchNorm2d.prior = 1
        return  outputs


    @torch.enable_grad() 
    def forward_and_adapt_memo(self, image, criterion):
        # self.model_per_image.eval()
        if self.prior_strength < 0:
            nn.BatchNorm2d.prior = 1
        else:
            nn.BatchNorm2d.prior = float(self.prior_strength) / float(self.prior_strength + 1)

        for iteration in range(self.niter):
            #just one image :(
            inputs = [aug(image) for _ in range(self.batch_size)]
            inputs = torch.stack(inputs).cuda()
            self.optimizer.zero_grad()
            outputs = self.model_per_image(inputs)
            loss, logits = criterion(outputs)
            loss.backward()
            self.optimizer.step()
        nn.BatchNorm2d.prior = 1

    def reset(self):
        self.model_per_image.load_state_dict(self.model_state_dict, strict=True)

def copy_model_and_optimizer(model, lr = 0.00025, weight_decay=0.0, prior_strength = 16):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_copy  = deepcopy(model)
    optimizer = optim.SGD(model_copy.parameters(), lr=lr, weight_decay=weight_decay)
    if prior_strength >= 0:
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
        nn.BatchNorm2d.forward = _modified_bn_forward
    # optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, model_copy,  optimizer #, optimizer_state

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

# https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)
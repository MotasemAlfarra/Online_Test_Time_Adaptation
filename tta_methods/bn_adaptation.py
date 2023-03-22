import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
# This code is adapted from the paper:
# https://proceedings.neurips.cc/paper/2020/file/85690f81aadc1749175c187784afc9ee-Paper.pdf
"Batch Size and prior strength N are set to 256 -> prior = 0.5"
class BN_Adaptation(nn.Module):
    """BN Adapts the model by updating the statistics of the BatchNorm Layers.
    """
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        prior = 0.5
        self.model = BayesianBatchNorm.adapt_model(self.model, prior)
        self.model.to(args.device)
    def forward(self, x):
        return self.model(x)

class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features, affine=False, momentum=1.0
        )

        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )
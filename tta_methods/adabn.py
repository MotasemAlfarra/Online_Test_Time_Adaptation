
import torch.nn as nn

# This code is adapted from the paper:
# https://arxiv.org/pdf/1603.04779.pdf
class AdaBn(nn.Module):
    """BN Adapts the model by updating the statistics of the BatchNorm Layers.
    """
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.model = configure_model(self.model)

    def forward(self, x):
        return self.model(x)



def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            # To force the model to use batch statistics
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

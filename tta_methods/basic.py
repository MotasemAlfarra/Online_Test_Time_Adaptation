import torch.nn as nn

class Basic_Wrapper(nn.Module):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, x):
        return self.model(x)   
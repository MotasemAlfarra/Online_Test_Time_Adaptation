import torch
import torch.nn as nn
from torchvision import transforms

from utils.dda_utils.image_adapt.diffusion import diffusion

class DDA(nn.Module):
    def __init__(self, model, args) -> None:
        super().__init__()

        self.model = model
        self.diffusion = diffusion(args.batch_size)
        self.preprocess = diffusion_preprocess()
        self.post_process = diffusion_post_process()

    def forward(self, x):
        output_original = self.model(x)
        diffused_x = self.diffusion(self.preprocess(x))
        output_diffused = self.model(self.post_process(diffused_x))
        return 0.5*(output_original + output_diffused)


class diffusion_preprocess(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trasnforms = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def forward(self, x):
        to_return = torch.zeros(x.shape[0],x.shape[1], 256, 256, device=x.device)
        for i in range(x.shape[0]):
            to_return[i] = self.trasnforms(x[i])
        return to_return

class diffusion_post_process(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trasnforms = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[2, 2, 2]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def forward(self, x):
        to_return = torch.zeros(x.shape[0],x.shape[1], 224, 224, device=x.device)
        for i in range(x.shape[0]):
            to_return[i] = self.trasnforms(x[i])
        return to_return
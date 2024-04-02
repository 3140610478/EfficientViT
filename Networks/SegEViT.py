import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor, nn

base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from efficientvit.seg_model_zoo import create_seg_model


class SegEViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.EViT = create_seg_model(
            name="b2", dataset="cityscapes", weight_url=os.path.abspath(os.path.join(base_folder, "./Networks/b2.pt"))
        )
        self.conv = nn.Conv2d(19, 1, 1)

    def forward(self, input: Tensor) -> Tensor:
        input_shape = input.shape[-2:]
        y = self.EViT.forward(input)
        y = F.interpolate(y, size=input_shape, mode="bilinear")
        return F.sigmoid(self.conv(y))

    def fit(self):
        self.conv.train()
        self.EViT.eval()
        self.EViT.requires_grad_(False)

    def train(self, *args, **kwargs):
        self.requires_grad_(True)
        super().train(*args, **kwargs)


if __name__ == "__main__":
    model = SegEViT()
    model.fit()
    pass

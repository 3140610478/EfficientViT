import os
import sys
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config

# train_folder = os.path.abspath(os.path.join(base_folder, "./Data/augmented_train"))
train_folder = os.path.abspath(os.path.join(base_folder, "./Data/train"))
val_folder = os.path.abspath(os.path.join(base_folder, "./Data/val"))
test_folder = os.path.abspath(os.path.join(base_folder, "./Data/test"))
# config.batch_size = 1


class ToDevice(torch.nn.Module):
    def __init__(self, device=config.device):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor):
        return input.to(self.device).detach()


# augmentation = None
augmentation = transforms.Compose([
    ToDevice(),
    transforms.RandomCrop(1024, padding=64, padding_mode="symmetric"),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
augmentation_complex = transforms.Compose([
    ToDevice(),
    transforms.CenterCrop(1020),
    transforms.Pad(600, padding_mode="reflect"),
    transforms.RandomAffine(
        degrees=15,
        shear=(-15, 15, -15, 15),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.CenterCrop(1024),
    transforms.RandomResizedCrop(
        1024, scale=(0.8, 1.0),
        ratio=(0.8, 1.25),
        antialias=True,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ElasticTransform(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


class Gaofen(Dataset):
    def __init__(self, data_folder, transform=ToDevice()):
        self.data_folder = os.path.abspath(data_folder)
        self.image, self.label, self.roi = [
            os.path.abspath(os.path.join(self.data_folder, "./image")),
            os.path.abspath(os.path.join(self.data_folder, "./label")),
            os.path.abspath(os.path.join(self.data_folder, "./roi")),
        ]
        self.ToTensor = transforms.ToTensor()
        self.transform = transform
        self.length = len([file for file in os.listdir(
            self.image) if file.endswith('.png')])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= self.length:
            raise StopIteration
        index += 1
        file_name = f"{index}.png"
        files = [
            os.path.abspath(os.path.join(
                self.data_folder, f"./image/{file_name}")),
            os.path.abspath(os.path.join(
                self.data_folder, f"./label/{file_name}")),
            os.path.abspath(os.path.join(
                self.data_folder, f"./roi/{file_name}")),
        ]
        image, label, roi = [self.ToTensor(Image.open(file)) for file in files]

        pack = torch.concatenate((image, label, roi), dim=0)
        pack = self.transform(pack).detach()
        image, label, roi = pack[:3], pack[3:4], pack[4:5]
        roi = roi.to(torch.int32)
        return image, label, roi


train_set = Gaofen(train_folder, augmentation)
val_set, test_set = Gaofen(val_folder), Gaofen(test_folder)


train_loader = DataLoader(train_set, config.batch_size, shuffle=True)
val_loader = DataLoader(val_set, config.batch_size, shuffle=True)
test_loader = DataLoader(test_set, config.batch_size, shuffle=True)
len_train, len_val, len_test = len(train_set), len(val_set), len(test_set)
info = f"""
Gaofen Datasets
1024x1024x3 images
1024x1024x1 labeles
"len_train, len_val = {len_train}, {len_val}
"""


if __name__ == "__main__":
    augmentation = transforms.ToTensor()
    inverse = transforms.ToPILImage()
    for (x, y, z) in train_loader:
        print(x.shape)
        print(y.shape)
        print(z.shape)
        pass

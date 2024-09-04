import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
import tifffile as tiff 
from torchvision import transforms

class SARMSI(Dataset):
    """
    num_classes: 10
    0) Background, 1) Tree, 2) Grassland, 3) Cropland, 4)
Low Vegetation, 5) Wetland, 6) Water, 7) Built-up, 8) Bare ground, 9) Snow.
    """
    CLASSES = ["Background", "Tree", "Grassland", "Cropland", "Low Vegetation", "Wetland", "Water", "Built-up", "Bare ground", "Snow"]

    PALETTE = torch.tensor([[0, 0, 0],
            [100, 40, 40],
            [55, 90, 80],
            [220, 20, 60],
            [153, 153, 153],
            [157, 234, 50],
            [128, 64, 128],
            [244, 35, 232],
            [107, 142, 35],
            [0, 0, 142],
            [102, 102, 156],
            [220, 220, 0],
            [70, 130, 180],
            [81, 0, 81],
            [150, 100, 100],
            ])
    
    def __init__(self, root: str = 'data/FMB', split: str = 'train', transform = None, modals = ['sar', 'msi0', 'msi1', 'msi2', 'msi3'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        if split == 'val':
            split = 'test'
        self.SAR = os.path.join(root,split,"SAR")
        self.MSI = os.path.join(root,split,"MSI")
        self.Label = os.path.join(root,split,"label")
        self.datasplit = os.path.join(root,f"{split}.txt")
        with open(self.datasplit, "r") as f:
            self.files = f.read().splitlines()
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        if not self.files:
            raise Exception(f"No images found in {self.datasplit}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def compose_sar(self, sar_data):
        assert sar_data.shape[0] == 2, "输入张量必须有两个通道"

        third_channel = sar_data.mean(dim=0, keepdim=True)  

        three_channel_tensor = torch.cat((sar_data, third_channel), dim=0)  
        return three_channel_tensor
    
    def compose_msi(self, msi_data):
        assert msi_data.shape[0] == 12, "输入张量必须有12个通道"
    
        split_tensors = torch.split(msi_data, 3, dim=0)
        
        return split_tensors
    
    def totensor(self, sar_data):
        sar_data = sar_data.astype(np.float32)

        H, W, C = sar_data.shape

        scaled_sar_data = np.zeros_like(sar_data)

        for c in range(C):  
            channel_data = sar_data[:, :, c]
            min_val = channel_data.min()
            max_val = channel_data.max()

            if max_val - min_val != 0:
                scaled_sar_data[:, :, c] = (channel_data - min_val) / (max_val - min_val)
            else:
                scaled_sar_data[:, :, c] = 0 
        t = transforms.ToTensor()
        sar_tensor = t(scaled_sar_data)

        return sar_tensor
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = self.files[index]
        SAR = os.path.join(self.SAR, item_name)
        MSI = os.path.join(self.MSI, item_name)
        Label = os.path.join(self.Label, item_name)

        sample = {}
        sample['sar'] = self.compose_sar(self.totensor(tiff.imread(SAR)))
        H, W = sample['sar'].shape[1:]
        msi_split = self.compose_msi(self.totensor(tiff.imread(MSI)))
        for idx in range(4):
            sample[f'msi{idx}'] = msi_split[idx]
            
        label = torch.from_numpy(tiff.imread(Label)).unsqueeze(0)
        label[label==255] = 0
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        # return sample, label, item_name
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)


if __name__ == '__main__':
    trainset = SARMSI('/media/hyh/1A24C88B24C86AF9/MMSeg-YREB')
    trainloader = DataLoader(trainset, batch_size=2, num_workers=4, drop_last=False, pin_memory=False)
    for sample, label in trainloader:
        print(sample)
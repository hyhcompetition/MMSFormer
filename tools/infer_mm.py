import numpy as np
import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF 
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import tifffile as tiff

def map_segmentation_to_rgb(mask, output_path):
    """
    将语义分割掩膜映射到 RGB 颜色并保存图像。

    参数:
    mask (numpy.ndarray): 输入的语义分割掩膜，形状为 (h, w)，值为 1-9。
    output_path (str): 保存RGB图像的路径。
    """
    # 定义一个颜色映射表，将每个类别（1-9）映射到一个 RGB 颜色
    color_map = {
        1: (255, 0, 0),    # 红色
        2: (0, 255, 0),    # 绿色
        3: (0, 0, 255),    # 蓝色
        4: (255, 255, 0),  # 黄色
        5: (255, 0, 255),  # 洋红
        6: (0, 255, 255),  # 青色
        7: (128, 128, 128),# 灰色
        8: (128, 0, 128),  # 紫色
        9: (0, 128, 128)   # 棕色
    }

    # 获取掩膜的高度和宽度
    h, w = mask.shape

    # 创建一个新的空的 RGB 图像（全黑）
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 遍历每个类别，并将对应的像素设置为指定颜色
    for category, color in color_map.items():
        rgb_image[mask == category] = color

    # 将 numpy 数组转换为 PIL 图像
    pil_image = Image.fromarray(rgb_image)

    # 保存图像到指定路径
    pil_image.save(output_path)
    print(f"RGB图像已保存到 {output_path}")
class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        # self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline_img = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        self.tf_pipeline_modal = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return pil_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
    
    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img
    def compose_sar(self, sar_data):
        assert sar_data.shape[0] == 2, "输入张量必须有两个通道"

        third_channel = sar_data.mean(dim=0, keepdim=True)  

        three_channel_tensor = torch.cat((sar_data, third_channel), dim=0)  
        return three_channel_tensor
    
    def compose_msi(self, msi_data):
        assert msi_data.shape[0] == 12, "输入张量必须有12个通道"
    
        split_tensors = torch.split(msi_data, 3, dim=0)
        
        return list(split_tensors)
    
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
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            x1 = img_fname.replace('/img', '/hha').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/lidar').replace('_rgb', '_lidar')
            x3 = img_fname.replace('/img', '/event').replace('_rgb', '_event')
            lbl_path = img_fname.replace('/img', '/semantic').replace('_rgb', '_semantic')
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            x1 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_hha'))
            x2 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_lidar'))
            x2 = x2.replace('.png', '_color.png')
            x3 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_event'))
            x3 = x3.replace('/image_00/data_rect/', '/').replace('.png', '_event_image.png')
            lbl_path = os.path.join(*[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])
        elif cfg['DATASET']['NAME'] == 'SARMSI':
            SAR = img_fname.replace('img', 'SAR')
            MSI = img_fname.replace('img', 'MSI')
            sar = self.compose_sar(self.totensor(tiff.imread(SAR)))
            msi_split = self.compose_msi(self.totensor(tiff.imread(MSI)))
            
        # image = io.read_image(img_fname)[:3, ...]
        # img = self.tf_pipeline_img(image).to(self.device)
        # # --- modals
        # x1 = self._open_img(x1)
        # x1 = self.tf_pipeline_modal(x1).to(self.device)
        # x2 = self._open_img(x2)
        # x2 = self.tf_pipeline_modal(x2).to(self.device)
        # x3 = self._open_img(x3)
        # x3 = self.tf_pipeline_modal(x3).to(self.device)


        sample = [sar]+msi_split
        sample = [x.unsqueeze(0).to(self.device) for x in sample]

        seg_map = self.model_forward(sample)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().to(int).numpy()
        # seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/sarmsi.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres', None]
    cases = ['lidarjitter']

    modals = cfg['DATASET']['MODALS']

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    # print(f"Model {cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}")
    # print(f"Model {cfg['DATASET']['NAME']}")

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results' / (cfg['DATASET']['NAME']+'_'+cfg['MODEL']['BACKBONE']+'_'+modals_name)
    
    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{str(test_file.stem)}.png")
    else:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', '*', 'val', '*', '*.png']))) # --- Deliver
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            source = os.path.join(test_file, 'val.txt')
            files = []
            with open(source) as f:
                files_ = f.readlines()
            for item in files_:
                file_name = item.strip()
                if ' ' in file_name:
                    # --- KITTI-360
                    file_name = os.path.join(*[str(test_file), file_name.split(' ')[0]])
                files.append(file_name)
        elif cfg['DATASET']['NAME'] == 'SARMSI':
            inference = os.path.join(test_file, 'test.txt')
            files=[]
            with open(inference, 'r') as f:
                files_ = f.read().splitlines()
            for fn in files_:
                files.append(os.path.join(test_file, 'test', 'img', fn))
        else:
            raise NotImplementedError()

        for file in files:
            print(file)
            # if not '2013_05_28_drive_0000_sync' in file:
            #     continue
            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            save_path = os.path.join(str(save_dir),file)
            map_segmentation_to_rgb(segmap, os.path.join('./vis',os.path.basename(file).replace('tif','png')))
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            segmap = segmap.astype(np.uint8)
            tiff.imsave(os.path.join('./predict',os.path.basename(file)), segmap)

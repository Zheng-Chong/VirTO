import os
import PIL
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from . import image_tools

# IUV:   1:躯干后 2:躯干前
#        3:右手 4:左手
#        5:左脚 6:右脚
#        7:右大腿后 8:左大腿后
#        9:右大腿前 10:左大腿前
#        11:右小腿后 12:左小腿后
#        13:右小腿前 14:左小腿前
#        15:左大臂内侧 16::右大臂内侧
#        17:左大臂外侧 18:右大臂外侧
#        19:左小臂内侧 20:右小臂内侧
#        21:左小臂外侧 22:右小臂外侧
#        23:右头部 24:左头部


# Image Transformer (resize image width to targeted resolution as Tensor)
def trans(img, resize=512, interp=PIL.Image.NEAREST):
    transformer = transforms.Compose([
        transforms.Resize(resize, interp),
        transforms.ToTensor()])
    return transformer(img)


# Dataset for training Parsing Generate Module)
class PGMDataset(Dataset):
    def __init__(self, root_dir, cloth_type, resize=256):
        self.resize = resize
        self.images = []
        for c in cloth_type:
            path = os.path.join(root_dir, c, 'densepose')
            for img in os.listdir(path):
                if img.endswith("IUV.png") and img.rfind('_back') == -1:
                    self.images.append(os.path.join(path, img[:img.rfind('_')]))
        print("Dataset Usable Pairs:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        # prefix: dir to cloth type
        # name: image name without postfix
        prefix, name = file[:file.rfind('/')-9], file[file.rfind('/')+1:]

        #   1. cloth masks (./cloth-mask/*-cloth_front.jpg)
        #   2. CIHP_PGN Parsing (./CIHP_PGN/parse/*.jpg)
        #   3. DensePose IUVs (./densepose/*_IUV.png')
        cloth_mask_dir = os.path.join(prefix, "cloth-mask", name[:name.rfind('-')] + "-cloth_front.jpg")
        target_seg = PIL.Image.open(os.path.join(prefix, "CIHP_PGN/parse", name+'.png'))
        target_iuv = PIL.Image.open(file + "_IUV.png")
        target_cloth_mask = PIL.Image.open(cloth_mask_dir)

        # Process and Integration data
        iuv = trans(target_iuv, resize=self.resize)[2, :, :].unsqueeze(0)
        cloth_mask = trans(target_cloth_mask, resize=self.resize)
        upper_seg = image_tools.extract_masks(trans(target_seg, resize=self.resize).view(self.resize, -1), [5, 7, 10, 14, 15])

        return upper_seg, iuv, cloth_mask

# from torch.utils.data import DataLoader
# root_dir = '/Users/fredrichie/Desktop/dataset/adidas_pre/men_t_shirt_pre'  # Path to preprocessed dataset
# dataset = CPMDataset(root_dir)
# train_data = DataLoader(dataset, batch_size=1, shuffle=True)
# for batch_data in train_data:
#     cloth_seg, divided_iuvs, target_cloth_seg = batch_data[0], batch_data[1], batch_data[2]
#     print(divided_iuvs.size())



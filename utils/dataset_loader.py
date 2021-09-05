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
    def __init__(self, root_dir, resize=256):
        self.resize = resize
        # root_dir: directory saving
        #   1. cloth masks (named 'cloth_mask')
        #   2. CIHP_PGN Parsing (named 'seg')
        #   3. DensePose IUVs (named 'IUV')
        self.cloth_mask_dir = os.path.join(root_dir, "cloth-mask")
        self.seg_dir = os.path.join(root_dir, "CIHP_PGN/parse")
        self.IUV_dir = os.path.join(root_dir, "densepose")

        self.images = []
        img_list = os.listdir(self.seg_dir)
        for img in img_list:
            if img.rfind("_vis") == -1 and img.endswith(".png"):
                self.images.append(img)

        print("Dataset Usable Pairs:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]

        target_seg = PIL.Image.open(os.path.join(self.seg_dir, file))
        target_iuv = PIL.Image.open(os.path.join(self.IUV_dir, file[:-4]+"_IUV.png"))
        target_cloth_mask = PIL.Image.open(os.path.join(self.cloth_mask_dir, file[:file.rfind('-')] + "-cloth_front.jpg"))

        # Process and Integration data
        iuv = trans(target_iuv, resize=self.resize)[2, :, :]
        cloth_mask = trans(target_cloth_mask, resize=self.resize)
        upper_seg = image_tools.extract_masks(trans(target_seg, resize=self.resize).view(self.resize, -1), [5, 7, 10, 14, 15])

        return upper_seg.unsqueeze(0), iuv.unsqueeze(0), cloth_mask

# from torch.utils.data import DataLoader
# root_dir = '/Users/fredrichie/Desktop/dataset/adidas_pre/men_t_shirt_pre'  # Path to preprocessed dataset
# dataset = CPMDataset(root_dir)
# train_data = DataLoader(dataset, batch_size=1, shuffle=True)
# for batch_data in train_data:
#     cloth_seg, divided_iuvs, target_cloth_seg = batch_data[0], batch_data[1], batch_data[2]
#     print(divided_iuvs.size())


# Dataset for training CGM(Cloth Generation Module)
class CGMDataset(Dataset):
    def __init__(self, root_dir):
        # root_dir: directory saving
        #   1.pictures(named 'pics')
        #   2.segmentation(named 'seg')
        #   3.IUVs(named 'IUV')
        self.pic_dir = os.path.join(root_dir, "pics")
        self.seg_dir = os.path.join(root_dir, "seg")

        self.images = []
        img_list = os.listdir(self.seg_dir)
        for img in img_list:
            if img.rfind("person") != -1:
                self.images.append(img)

        print("Usable Data Number:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        resize = 256
        # A piece of training data contains:
        #   1.cloth image;
        #   2.target Upper seg;
        #   3.target Upper image.
        target_img = PIL.Image.open(os.path.join(self.pic_dir, file[:-3]+"jpg"))
        target_img = trans(target_img, resize=resize)
        target_seg = PIL.Image.open(os.path.join(self.seg_dir, file[:-3]+"png"))
        target_seg = trans(target_seg, resize=resize)
        prefix = file[0:file.rfind('-')]
        cloth = PIL.Image.open(os.path.join(self.pic_dir, prefix+"-cloth_front.jpg"))
        cloth = trans(cloth, resize=resize)

        # Process and Integration data
        target_cloth_seg = target_seg.masked_fill(mask=target_seg == 5 / 255, value=torch.tensor(1))
        target_cloth_seg = image_tools.binarize_image(target_cloth_seg, 0.99, 1)
        ground_truth = target_img.masked_fill(mask=torch.cat([target_cloth_seg, target_cloth_seg, target_cloth_seg], 0) == 0, value=torch.tensor(0))

        return cloth, target_cloth_seg, ground_truth
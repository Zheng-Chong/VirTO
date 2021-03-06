import os
import torch
import PIL
from torchvision import transforms

# Image Transformer (resize image width to 512 pixels and to Tensor)
trans = transforms.Compose([
    transforms.Resize(512, PIL.Image.NEAREST),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()


# Save Tensor to Target Directory
def save_image(img, name, dst):
    image = img.cpu().clone()  # Clone the tensor to not do changes on it
    image = unloader(image)
    if not os.path.exists(dst):
        os.makedirs(dst)
    save_path = os.path.join(dst, name+".png")
    image.save(save_path)
    # print('Save Image to %s' % save_path)


# Auto-grayscale tensor
def grayscale_tensor(tensor):
    if len(list(tensor.size())) == 3:
        cs = []
        for index, i in enumerate(tensor):
            channel = i.clone()
            channel = channel.masked_fill(mask=channel > 0, value=torch.tensor((index+1)/tensor.size(0)))
            cs.append(channel)
        res = torch.stack(cs, 0).sum(0).unsqueeze(0)
        return res
    else:
        print("ERROR: only tensor in 3 dimensions can be visualized!")
        return


# Divide Different Mask Area Into Different Channels
def extract_masks(tensor, mask_values, merge=False):
    channels = []
    for v in mask_values:
        v = v/255
        color = v if merge else 1
        channel = tensor.clone()
        channel = channel.masked_fill(mask=channel != v, value=torch.tensor(0))
        channels.append(channel.masked_fill(mask=channel == v, value=torch.tensor(color)))
    res = torch.stack(channels, 0)
    if merge:
        res = res.sum(0).unsqueeze(0)
    return res


def splice_image(images, grayscales=False):
    if grayscales:
        res = torch.cat(images, 1)
        return res * 20
    res = torch.cat(images, 2)
    return res


# Unify the non-zero values of the image to ‘value’
def binarize_image(image, division, value):
    image = image.masked_fill(mask=image > division, value=torch.tensor(value))
    image = image.masked_fill(mask=image != value, value=torch.tensor(0))
    return image


def select_value(image, value):
    mask = image != value
    image = image.masked_fill(mask=mask, value=torch.tensor(0))
    return image


# target_img = PIL.Image.open("/Users/fredrichie/PycharmProjects/MyTryOn2.0/test_iuv.png")
# target_img = trans(target_img)[2]
# target_img = extract_masks(target_img, list(range(1, 25)))
# print(target_img.size())
# visualize_tensor(target_img, "test2", "/Users/fredrichie/PycharmProjects/MyTryOn2.0/visualization")

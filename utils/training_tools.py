import torch
import time
from utils import image_tools

def save_model(epoch, model, optimizer, name):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './checkpoints/%s.pth' % name)
    # print("Model state saved to ./checkpoints/%s.pth !" % name)


def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad


def record_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def training_log(epoch, batch, time, losses, freq):
    info = 'Epoch %3i Batch %4i, time: %3f s ' % (epoch, batch, time)
    for key in losses.keys():
        info += ', %s:%4f' % (key, losses[key]/freq)
        losses[key] = 0
    print(info)


def visualize(imgs, name, dir):
    grayscales = False if imgs[0].size(0) == 3 else True
    result = image_tools.splice_image(imgs, grayscales=grayscales)
    image_tools.save_image(result, name, dir)
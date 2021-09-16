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


def set_requires_grad(net1, net2):
    if net1 is not None:
        for param in net1.parameters():
            param.requires_grad = True
    if net2 is not None:
        for param in net2.parameters():
            param.requires_grad = False


def record_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def training_log(epoch, batch, time, losses, freq, lr=None, txt_log=None):
    info = 'Epoch %3i Batch %4i, time: %3f s ' % (epoch, batch, time)
    for key in losses.keys():
        info += ', %s:%4f' % (key, losses[key]/freq)
        losses[key] = 0
    if lr is not None:
        info += ", lr=%f6" % lr
    print(info)
    if txt_log is not None:
        with open("test.txt", "a") as f:
            f.write(info)


def visualize(imgs, name, dir):
    vis = []
    for i in imgs:
        if i.size(0) != 3:
            vis.append(image_tools.grayscale_tensor(i))
        else:
            vis.append(i[0].unsqueeze(0))
    result = torch.cat(vis, 2)
    image_tools.save_image(result, name, dir)
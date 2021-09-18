import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import dataset_loader, training_tools
from models import networks

from losses import perceptual_loss as PLoss

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = "CMG_training_log.txt"

''' 
    Hyper-parameter
'''
batch_size = 8
patch_size = 64
resize = 256
epoch_num = 100
clip_max_norm = 5
weight_adv, weight_l1, weight_per, weight_BDR = 1, 1.5, 2, 0.2

''' 
    Training parameters
'''
model_name = "CMG-UNet-1.0.2"  # The name to tore checkpoint
save_frequency = 20  # The number of intervals between storage of checkpoints
continue_training = True  # Whether to find & use pre-training checkpoint

root_dir = '/home/lingfeimo/cz/Dataset/adidas/men'  # Path to preprocessed dataset
cloth_type = ['t_shirt', 'coat', 'vest', 'waistcoat']

if sys.platform == 'darwin':
    root_dir = '/Users/fredrichie/Desktop/dataset/adidas_pre/men'
    cloth_type = ['test']
    save_frequency = 1


G_dir = './checkpoints/%s.pth' % model_name  # Path to pre-trained Generator checkpoint


def train(model_name='default', epoch_num=500, save_frequency=100, resize=256, patch_size=128, visualize=True, continue_training=True):
    # Initial dataset
    dataset = dataset_loader.CMGDataset(root_dir, cloth_type, resize=resize)
    # Setup Cloth Parsing Module Network
    G, D = networks.CMGenerator(in_channels=3), networks.Discriminator(1)
    start_epoch = 0
    if os.path.isfile(G_dir) and continue_training:
        print("Load state dict from %s ...." % G_dir)
        start_epoch = torch.load(G_dir, map_location=device)['epoch']
        G.load_state_dict(torch.load(G_dir, map_location=device)['model_state_dict'])
    G.to(device)
    D.to(device)
    # Setup Optimizer
    optimizer_g = torch.optim.RMSprop(G.parameters(), lr=0.06)
    optimizer_d = torch.optim.RMSprop(D.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, T_0=5, T_mult=2)

    # Train for epochs
    for epoch in range(start_epoch, start_epoch + epoch_num):

        train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Load dataset
        G.train()  # switch to train mode
        loss_record = {'S-L1': 0, 'Adv': 0, 'Dis': 0}

        for i_batch, batch_data in enumerate(train_data):
            cloth_img, groud_truth = batch_data[0].to(device), batch_data[1].to(device)
            # Record Start Time
            start = training_tools.record_time()
            # Generate fake result
            pred_seg = G(cloth_img)
            # Calculate Losses
            adv_criterion = networks.AdversarialLoss(lsgan=True)
            bce_loss = nn.SmoothL1Loss()(pred_seg, groud_truth)
            gen_loss = adv_criterion(pred_seg, D, patch_size, True)
            dis_loss = adv_criterion(pred_seg, D, patch_size, False, groud_truth)

            loss_g = weight_l1 * bce_loss + weight_adv * gen_loss
            loss_d = weight_adv * dis_loss

            # Record Losses
            loss_record['S-L1'] += float(bce_loss)
            loss_record['Adv'] += float(gen_loss)
            loss_record['Dis'] += float(dis_loss)

            # Update generator's weights
            training_tools.set_requires_grad(None, D)
            G.zero_grad()
            loss_g.backward(retain_graph=True)
            scheduler.step(epoch + i_batch // len(train_data))

            # Update discriminator's weights
            training_tools.set_requires_grad(D, None)
            D.zero_grad()
            loss_d.backward()
            optimizer_g.step()
            # Clipping Gradient
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=clip_max_norm, norm_type=2)
            optimizer_d.step()

            # Record End Time
            end = training_tools.record_time()
            time_cost = end - start

            # Console training info per x batches
            if (i_batch + 1) % save_frequency == 0:
                # Visualization
                if visualize:
                    training_tools.visualize([cloth_img[0], groud_truth[0], pred_seg[0]],
                                             '%s-e%i-b%i' % (model_name, epoch, i_batch+1), "./visualization")
                # Save checkpoint
                training_tools.save_model(epoch, G, optimizer_g, model_name)
                # Training info log
                training_tools.training_log(epoch, i_batch+1, time_cost, loss_record, save_frequency,
                                            optimizer_g.param_groups[-1]['lr'], txt_log=out)

        # Save Checkpoint for each epoch
        training_tools.save_model(epoch, G, optimizer_g, model_name)


train(model_name, epoch_num, save_frequency, resize, patch_size, True, continue_training)

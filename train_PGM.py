import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import dataset_loader, training_tools, image_tools
from models import networks

from losses import perceptual_loss as PLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = open("PGM_training_log.txt", 'w')

''' 
    Hyper-parameter
'''
batch_size = 16
patch_size = 64
epoch_num = 1000
weight_adv, weight_l1, weight_per, weight_BDR = 1, 1, 2, 0.2

''' 
    Training parameters
'''
model_name = "PGM-XCeption-1.0"  # The name to tore checkpoint
save_frequency = 50  # The number of intervals between storage of checkpoints
continue_training = True  # Whether to find & use pre-training checkpoint

root_dir = '/home/lingfeimo/cz/Dataset/men/t_shirt'  # Path to preprocessed dataset
G_dir = './checkpoints/%s.pth' % model_name  # Path to pre-trained Generator checkpoint


def train(model_name='default', epoch_num=500, save_frequency=100, patch_size=128, visualize=True, continue_training=True):
    # Initial dataset
    dataset = dataset_loader.PGMDataset(root_dir)
    # Setup Cloth Parsing Module Network
    G, start_epoch = networks.PGMGenerator(), 0
    if os.path.isfile(G_dir) and continue_training:
        print("Load state dict from %s ...." % G_dir)
        start_epoch = torch.load(G_dir, map_location=device)['epoch']
        G.load_state_dict(torch.load(G_dir, map_location=device)['model_state_dict'])
    G.to(device)
    # Setup Optimizer
    optimizer_g = torch.optim.Adam(G.parameters())

    # Train for epochs
    for epoch in range(start_epoch, start_epoch + epoch_num):

        train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Load dataset
        G.train()  # switch to train mode
        loss_record = {'S-L1': 0, 'Pecp': 0}

        for i_batch, batch_data in enumerate(train_data):
            groud_truth, iuv, cloth_mask = batch_data[0].to(device), batch_data[1].to(device), \
                                                        batch_data[2].to(device)
            # Record Start Time
            start = training_tools.record_time()

            # Generate fake result
            pred_seg = G.forward(iuv, cloth_mask)

            # Calculate Losses
            smooth_l1_loss = nn.SmoothL1Loss()(pred_seg, groud_truth)
            per_loss = PLoss.PerceptualLoss134()(pred_seg.repeat_interleave(repeats=3, dim=1),
                                                 groud_truth.repeat_interleave(repeats=3, dim=1))
            loss_g = weight_l1 * smooth_l1_loss + weight_per * per_loss


            # Record Losses
            loss_record['S-L1'] += float(smooth_l1_loss)
            loss_record['Pecp'] += float(per_loss)

            # Update generator's weights
            loss_g.backward(retain_graph=True)
            optimizer_g.step()

            # Record End Time
            end = training_tools.record_time()
            time_cost = end - start

            # Console training info per x batches
            if (i_batch + 1) % save_frequency == 0:
                # Visualization
                if visualize:
                    training_tools.visualize([iuv[0][0], groud_truth[0][0], pred_seg[0][0]],
                                             '%s-e%i-b%i' % (model_name, epoch, i_batch+1), "./visualization")
                # Save checkpoint
                training_tools.save_model(epoch, G, optimizer_g, model_name)
                # Training info log
                training_tools.training_log(epoch, i_batch+1, time_cost, loss_record, save_frequency)

        # Save Checkpoint for each epoch
        training_tools.save_model(epoch, G, optimizer_g, model_name)


train(model_name, epoch_num, save_frequency, patch_size, True, continue_training)

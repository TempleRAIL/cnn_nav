#!/usr/bin/env python
#
# file: $ISIP_EXP/so-scope/scripts/train.py
#
# revision history: xzt
#  20200724 (TE): first version
#
# usage:
#
# This script hold the training code
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# visualize:
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# import the model and all of its variables/functions
#
from model import *

# import modules
#
import sys
import os


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
model_dir = './model/model.pth'  # the path of model storage 
NUM_ARGS = 3
NUM_EPOCHS = 8000 
BATCH_SIZE = 128 
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# for reproducibility, we seed the rng
#
set_seed(SEED1)       

# adjust_learning_rate
#　
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-3
    if epoch > 50:
        lr = 3e-4
    if epoch > 200:
        lr = 3e-5
    if epoch > 21000:
        lr = 1e-5
    if epoch > 32984:
        lr = 1e-6
    if epoch > 48000:
       # lr = 5e-8
       lr = lr * (0.1 ** (epoch // 110000))
    #  if epoch > 8300:
    #      lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train function:
def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    ################################## Train #####################################
    # Set model to training mode
    model.train()  
    # for each batch in increments of batch size
    #
    running_loss = 0
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scan_maps = batch['scan_map']
        scan_maps = scan_maps.to(device)
        ped_maps = batch['ped_map']
        ped_maps = ped_maps.to(device)
        sub_goals = batch['sub_goal']
        sub_goals = sub_goals.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)

        # set all gradients to 0:
        optimizer.zero_grad()
        # feed the network the batch
        #
        output = model(ped_maps, scan_maps, sub_goals)
        #writer.add_graph(model,[batch_ped_pos_t, batch_scan_t, batch_goal_t])    
        # get the loss
        #
        loss = criterion(output, velocities)
        # perform back propagation:
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        # get the loss:
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  

        running_loss += loss.item()
        
        # display informational message
        #
        if(i % 1280 == 0):
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                    .format(epoch, epochs, i + 1, num_batches, loss.item()))

    train_loss = running_loss / len(dataset) #counter 

    return train_loss

# validate function:
def validate(model, dataloader, dataset, device, criterion):
    ################################## Train #####################################
    # set model to evaluation mode:
    model.eval()
    # for each batch in increments of batch size
    #
    running_loss = 0
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scan_maps = batch['scan_map']
        scan_maps = scan_maps.to(device)
        ped_maps = batch['ped_map']
        ped_maps = ped_maps.to(device)
        sub_goals = batch['sub_goal']
        sub_goals = sub_goals.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)

        # feed the network the batch
        #
        output = model(ped_maps, scan_maps, sub_goals)
        #writer.add_graph(model,[batch_ped_pos_t, batch_scan_t, batch_goal_t])    
        # get the loss
        #
        loss = criterion(output, velocities)
        # get the loss:
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  

        running_loss += loss.item()

    val_loss = running_loss / len(dataset) #counter 

    return val_loss

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main(argv):

    # ensure we have the correct amount of arguments
    #
    #global cur_batch_win
    if(len(argv) != NUM_ARGS):
        print("usage: python nedc_train_mdl.py [MDL_PATH] [TRAIN_PATH] [DEV_PATH]")
        exit(-1)

    # define local variables
    #
    mdl_path = argv[0]
    pTrain = argv[1]
    pDev = argv[2]

    # get the output directory name
    #
    odir = os.path.dirname(mdl_path)

    # if the odir doesn't exits, we make it
    #
    if not os.path.exists(odir):
        os.makedirs(odir)

    # set the device to use GPU if available
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### train:
    print('...Start reading data...')
    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    #[ped_pos_t, scan_t, goal_t, vel_t] = get_data(pTrain)
    train_dataset = NavDataset(pTrain, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, \
                                                   shuffle=True, drop_last=True, pin_memory=True)
    #train_data = train_data - np.mean(train_data, axis=0)
    
    ### dev:

    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    #[ped_pos_d, scan_d, goal_d, vel_d] = get_data(pDev)
    dev_dataset = NavDataset(pDev, 'dev')
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, \
                                                   shuffle=True, drop_last=True, pin_memory=True)
    #dev_data = dev_data - np.mean(dev_data, axis=0)
    print('...Finish reading data...')

    # instantiate a model
    #
    model = CNN(Bottleneck, [2, 1, 1])

    # moves the model to device (cpu in our case so no change)
    #
    model.to(device)

    # set the adam optimizer parameters
    #
    opt_params = { LEARNING_RATE: 0.001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }

    # set the loss and optimizer
    #
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)

    # create an optimizer, and pass the model params to it
    #
    optimizer = Adam(model.parameters(), **opt_params)

    # get the number of epochs to train on
    #
    epochs = NUM_EPOCHS
    
    # if there are trained models, continue training:
    if os.path.exists(mdl_path):
        checkpoint = torch.load(mdl_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load epoch {} success'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models, restart training')

    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])

    # moves the model to device (cpu in our case so no change)
    #
    model.to(device)

    # tensorboard writer:
    writer = SummaryWriter('runs')

    # for each epoch
    #
    #loss_train = []
    #loss_vector = []
    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):

        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #
        train_epoch_loss = train(
            model, train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs
        )
        
        ################################## Test #####################################
        valid_epoch_loss = validate(
            model, dev_dataloader, dev_dataset, device, criterion
        )

        # log the epoch loss
        writer.add_scalar('training loss',
                        train_epoch_loss,
                        epoch)
        writer.add_scalar('validation loss',
                        valid_epoch_loss,
                        epoch)

        print('Train set: Average loss: {:.4f}'.format(train_epoch_loss))
        print('Validation set: Average loss: {:.4f}'.format(valid_epoch_loss))

        # save the model
        #
        if(epoch % 100 == 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='./model/model' + str(epoch) +'.pth'
            torch.save(state, path)
        
        epoch_num = epoch

    # save the final model
    if torch.cuda.device_count() > 1: # multiple GPUS: 
        state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    else:
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    torch.save(state, mdl_path)

    # exit gracefully
    #

    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file

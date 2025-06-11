#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/decode.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#  python decode.py odir mfile data
#
# arguments:
#  odir: the directory where the hypotheses will be stored
#  mfile: input model file
#  data: the input data list to be decoded
#
# This script decodes data using a simple MLP model.
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
from tqdm import tqdm
import numpy as np

# import the model and all of its variables/functions
#
from model import *

from collections import OrderedDict

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
NUM_ARGS = 3
SPACE = " "
HYP_EXT = ".hyp" 
GRT_EXT = ".grt"           
log_dir = '../model/model.pth'   

# for reproducibility, we seed the rng
#
set_seed(SEED1)            

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

    # ensure we have the correct number of arguments
    #
    if(len(argv) != NUM_ARGS):
        print("usage: python nedc_decode_mdl.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    # define local variables
    #
    odir = argv[0]
    mdl_path = argv[1]
    fname = argv[2]

    # if the odir doesn't exist, we make it
    #
    if not os.path.exists(odir):
        os.makedirs(odir)

    # get the hyp file name
    #
    hyp_name = os.path.splitext(os.path.basename(fname))[0] + HYP_EXT
    grt_name = os.path.splitext(os.path.basename(fname))[0] + GRT_EXT

    # set the device to use GPU if available
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    #[ped_pos_e, scan_e, goal_e, vel_e] = get_data(fname)
    eval_dataset = NavDataset(fname, 'test')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    # instantiate a model
    #
    model = CNN(Bottleneck, [2, 1, 1])

    # moves the model to the device
    #
    model.to(device)

    # set the model to evaluate
    #
    model.eval()

    # set the loss and optimizer
    #
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)

    # load the weights
    #
    checkpoint = torch.load(mdl_path, map_location=device)
    '''
    # create new OrderedDict that does not contain 'module.'
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove 'module.'
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    '''
    model.load_state_dict(checkpoint['model'])

    # the output file
    #
    try:
        ofile = open(os.path.join(odir, hyp_name), 'w+')
        vel_file = open(os.path.join(odir, grt_name), 'w+')
    except IOError as e:
        print(os.path.join(odir, hyp_name))
        print("[%s]: %s" % (hyp_name, e.strerror))
        exit(-1)

    # get the number of data points
    #
    num_points = len(eval_dataset)

    # for each data point
    #
    running_loss = 0
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        #
        print("decoding %4d out of %d" % (i+1, num_points))    

        # collect the samples as a batch:
        scan_maps = batch['scan_map']
        scan_maps = scan_maps.to(device)
        sub_goals = batch['sub_goal']
        sub_goals = sub_goals.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)

        # feed the network the batch
        #
        output = model(scan_maps, sub_goals)
        #writer.add_graph(model,[batch_ped_pos_t, batch_scan_t, batch_goal_t])    
        # get the loss
        #
        loss = criterion(output, velocities)
        # get the loss:
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  

        running_loss += loss.item()
        # write the highest probablity to the file
        #
        ofile.write(str(float(output.data.cpu().numpy()[0,0])) + \
                    SPACE + str(float(output.data.cpu().numpy()[0,1])) + NEW_LINE)
        vel_file.write(str(float(velocities[0,0])) + \
                    SPACE + str(float(velocities[0,1])) + NEW_LINE)

    # loss:
    val_loss = running_loss / counter 
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    # close the file
    #
    ofile.close()
    vel_file.close()
    
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

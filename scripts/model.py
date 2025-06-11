#!/usr/bin/env python
#
# file: $ISIP_EXP/so-scope/scripts/model.py
#
# revision history: xzt
#  20200724 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
import numpy as np
import numpy.matlib

# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
NEW_LINE = "\n"

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
POINTS = 1080
IMG_SIZE = 80
SEQ_LEN = 10
class NavDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        self.LASER_CLIP = 30
        self.mu_ped_pos = -0.0001 
        self.std_ped_pos = 0.0391 
        self.mu_scan = 5.3850 
        self.std_scan = 4.2161 
        # read the names of image data:
        self.scan_file_names = []
        self.ped_file_names = []
        self.goal_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans_base/'+file_name+'.txt', 'r')
        fp_ped = open(img_path+'/peds_local/'+file_name+'.txt', 'r')
        fp_goal = open(img_path+'/sub_goals_local/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans_base/'+line)
        for line in fp_ped.read().split(NEW_LINE):
            if('.npy' in line): 
                self.ped_file_names.append(img_path+'/peds_local/'+line)
        for line in fp_goal.read().split(NEW_LINE):
            if('.npy' in line): 
                self.goal_file_names.append(img_path+'/sub_goals_local/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_scan.close()
        fp_ped.close()
        fp_goal.close()
        fp_vel.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        if(idx+(SEQ_LEN) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN)

        # create lidar historical map:
        scan_avg = np.zeros((20,IMG_SIZE))
        for n in range(SEQ_LEN):
            # get the scan data:
            scan_name = self.scan_file_names[idx_s+n]
            scan = np.load(scan_name)
            scan_tmp = scan[180:-180]
            for i in range(IMG_SIZE):
                scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg,1,4)
        scan_map = scan_avg_map.reshape(6400)

        # create pedestrian kinematic maps:
        tracked_ped_name = self.ped_file_names[idx_s + SEQ_LEN-1]
        tracked_peds = np.load(tracked_ped_name)      # tracked pedestrians
        ped_map = np.zeros((2,80,80))  # cartesian velocity map
        for tracked_ped in tracked_peds:
            # relative positions and velocities:
            # position:
            x = tracked_ped[0] 
            y = tracked_ped[1] 
            # velocity:
            vx = tracked_ped[2]
            vy = tracked_ped[3]
            # 20m * 20m occupancy map:
            if(x >= 0 and x <= 20 and np.abs(y) <= 10):
                # bin size: 0.25 m
                c = int(np.floor(-(y-10)/0.25))
                r = int(np.floor(x/0.25))

                if(r == 80):
                    r = r - 1
                if(c == 80):
                    c = c - 1
                # cartesian velocity map
                ped_map[0,r,c] = vx
                ped_map[1,r,c] = vy

        # Normalization: scan_map
        scan_map = (scan_map - self.mu_scan) / self.std_scan

        # Normalization: ped_map
        ped_map = (ped_map - self.mu_ped_pos) / self.std_ped_pos

        # get the sub goal data:
        goal_name = self.goal_file_names[idx_s + SEQ_LEN-1]
        sub_goal = np.load(goal_name)

        # get the velocity data:
        vel_name = self.vel_file_names[idx_s + SEQ_LEN-1]
        vel = np.load(vel_name)
        
        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scan_map)
        ped_tensor = torch.FloatTensor(ped_map)
        sub_goal_tensor = torch.FloatTensor(sub_goal)
        vel_tensor =  torch.FloatTensor(vel)

        data = {
                'scan_map': scan_tensor,
                'ped_map': ped_tensor,
                'sub_goal': sub_goal_tensor,
                'velocity': vel_tensor, 
                }

        return data

#
# end of function


#------------------------------------------------------------------------------
#
# ResNet blocks
#
#------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2 #4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#
# end of ResNet blocks


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch MLP model
#
class CNN(nn.Module):

    # function: init
    #
    # arguments: input_size - int representing size of input
    #            hidden_size - number of nodes in the hidden layer
    #            num_classes - number of classes to classify
    #
    # return: none
    #
    # This method is the main function.
    #
    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):

        # inherit the superclass properties/methods
        #
        super(CNN, self).__init__()
        # define the model
        #
        ################## ped_pos net model: ###################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion + 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)           

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, ped_pos, scan, goal):
        ###### Start of fusion net ######
        ped_in = ped_pos.reshape(-1,2,80,80)
        scan_in = scan.reshape(-1,1,80,80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)
        
        # See note [TorchScript super()]
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)

        identity2 = self.downsample2(x)

        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)


        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)
        ###### End of fusion net ######

        ###### Start of goal net #######
        goal_in = goal.reshape(-1,2)
        goal_out = torch.flatten(goal_in, 1)
        ###### End of goal net #######
        # Combine
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.fc(fc_in)  

        return x

    def forward(self, ped_pos, scan, goal):
        return self._forward_impl(ped_pos, scan, goal)
    #
    # end of method
#
# end of class

#
# end of file

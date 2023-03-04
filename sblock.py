import os.path as osp
import math

import numpy as np
from numpy.core.defchararray import _binary_op_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F

import nevergrad as ng

import r1
import functions as f


#The Structure Block is a differentiable, convolutional Pytorch layer, which activates when it observes a set of
#features arranged in a specific spatial structure.
# 
#It is a 3D convolution-like block of size N x P x P, where N = number of features in previous layer, and P = perceptual field size.
#A max filter is applied at each P x P slice to constrain activation to a single pixel/location in the image
#Upsampling of the P x P field can be performed to apply the block to a larger region in the image.


# Forward Algorithm
# The input to this layer is a tensor of size (H x W x N)
# Apply in Order:
# (1) N FH x FW convolutions
# (2) Max Function to each
# (3) MLP with N inputs and 1 output
# Do this M times for the number of features/nodes in this layer, M
# Do this for each pixel in the (zero-padded) image.
# The output of this layer should be a tensor of size (H x W x M)

#Backward Algorithm


#class Conv_Edge_Detector(torch.nn.Module):



class MLP(torch.nn.Module):

    def __init__(self, input_size, mlpnodes, batchnorm):

        super(MLP, self).__init__()

        self.nodes = mlpnodes
        self.numlayers = len(self.nodes)
        self.fcs = nn.ModuleList()

        for idx in range(self.numlayers):

            if idx == 0:
                self.fcs.append(torch.nn.Linear(input_size, self.nodes[idx]))
            else:
                self.fcs.append(torch.nn.Linear(self.nodes[idx-1], self.nodes[idx]))

        self.activation = torch.nn.ReLU()
        self.use_bn = batchnorm
        self.bns = []

        if self.use_bn == True:
            for n in mlpnodes:
                self.bns.append(nn.BatchNorm2d(n))
        

    def forward(self, input):

        x = input 

        for idx in range(self.numlayers):

            x = self.fcs[idx](x)

            if self.use_bn == True:
                x = self.bns[idx](x)
                
            x = self.activation(x)
        
        out = x

        return out


class StructFun(torch.nn.Module):

    def __init__(self, fh, fw, fd, initprocedure, snnopts):
        
        super(StructFun, self).__init__()

        #Create differentiable 3D parameter block, S
        self.S = torch.empty(1, fd, fh, fw, dtype=torch.float).to('cuda')
        self.S.requires_grad = True
        
        #self.S = nn.Parameter(torch.empty(1, fd, fh, fw), requires_grad=True).cuda()
        #self.S = torch.empty(1, fd, fh, fw) #S is a Tensor of size [fh x fw x fd]
        #self.S.requires_grad = True

        #Set Tensor dimensions

        self.field_depth = fd
        self.field_height = fh
        self.field_width = fw

        self.N = -1
        self.D = -1
        self.H = -1
        self.W = -1

        #Set hyperparameters: kernel size, upsampling transform, and unfolding transform

        self.use_var_weighting = snnopts['use_var_weighting']
        self.k1 = snnopts['scale']
        self.k2 = snnopts['shift']

        #self.scaled_kernel_height = 2*math.floor(fh*sf/2) + 1
        #self.scaled_kernel_width = 2*math.floor(fw*sf/2) + 1

        #self.kernelsize = (self.scaled_kernel_height, self.scaled_kernel_width)

        #self.up = nn.Upsample(size = self.kernelsize, mode='nearest')
        #self.unfold = nn.Unfold(kernel_size = self.kernelsize, padding = ( math.floor(self.kernelsize[0]/2), math.floor(self.kernelsize[1]/2) ) )

        #Initialize SBlock weights
        if initprocedure == 'xavier_uniform':
            self.S = torch.nn.init.xavier_uniform_(self.S, gain=1.0)
        elif initprocedure == 'xavier_normal':
            print('The initialization procedure is not known, or has not been implemented.')
            print('Available initialization options are:' + ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'])
            #raise NotImplementedError


    def forward(self, input, bias=None):

        self.N = input.shape[0]
        self.D = input.shape[1]
        self.H = input.shape[2]
        self.W = input.shape[3]

        #Sup = self.up(self.S) #Replace upsampling with fractional maxpooling

        Sflat = torch.flatten(self.S, start_dim = 2, end_dim = -1)

        Sflat = torch.unsqueeze(Sflat, 3)

        U = self.pool(input)

        U = self.unfold(U)

        U = torch.reshape(U, (self.N, self.D, Sflat.shape[2], U.shape[2]))

        vals = Sflat*U #Do we want a convolution or to apply SR directly to each layer? #

        if self.use_var_weighting == True:

            vals = f.InvVarWeighting(vals, self.field_height, self.field_width, self.k1, self.k2) * vals

        M = torch.sum(vals, 2, keepdim=True) #Should we do Maxpool on the input, or just upsample the field block, S? #Change back to torch.max?


        return M


    def set_transforms(self, sf, Him, Wim):

        self.scale_factor = sf
        self.scaled_img_height = math.floor(Him/sf) #Scale the input size by dividing the Image Height/Width by the scale factor of this layer
        self.scaled_img_width = math.floor(Wim/sf)
        #self.scaled_kernel_height = 2*math.floor(self.field_height*self.scale_factor/2) + 1
        #self.scaled_kernel_width = 2*math.floor(self.field_width*self.scale_factor/2) + 1

        self.kernelsize = (self.field_height, self.field_width)
        #self.kernelsize = (self.scaled_kernel_height, self.scaled_kernel_width)

        #self.up = nn.Upsample(size = self.kernelsize, mode='nearest')
        self.unfold = nn.Unfold(kernel_size = self.kernelsize, padding = ( math.floor(self.kernelsize[0]/2), math.floor(self.kernelsize[1]/2) ) )
        self.pool = nn.FractionalMaxPool2d((3, 3), output_size = (self.scaled_img_height, self.scaled_img_width))

        return True
        


class StructBlock(nn.Module):

    def __init__(self, SB, FC, batchnorm):
        super(StructBlock, self).__init__()
        self.SBlock = SB
        self.mlp = FC

        self.use_bn = False
        self.bn = None
        if batchnorm == True:
            self.use_bn = True
            self.bn = nn.BatchNorm2d(self.SBlock.field_depth)
        



    def forward(self, input, bias=None, initial=0):

        #h = input.shape[2]
        #w = input.shape[3]
        ##fold = nn.Fold(output_size = (h, w), kernel_size = (1, 1))

        hout = self.SBlock.scaled_img_height
        wout = self.SBlock.scaled_img_width

        M = self.SBlock(input, bias)

        M = torch.reshape(M, (M.shape[0], M.shape[1], hout, wout))

        if self.use_bn == True:

            M = self.bn(M)

        M = M.permute(0, 2, 3, 1)

        M = self.mlp(M)

        return M

     



class StructLayer(nn.Module):

    def __init__(self, in_channels, out_channels, snnopts, stride = (1, 1)):

        super(StructLayer, self).__init__()

        self.numnodes = out_channels
        self.initprocedure = snnopts['init_procedure']
        self.fh = snnopts['field_height'] #Field Height
        self.fw = snnopts['field_width'] #Field Width
        self.fd = in_channels #Field Depth = Number of Channels in output of previous layer
        self.SBlocks = torch.nn.ModuleList()

        for i in range(self.numnodes):
            sb = StructFun(self.fh, self.fw, self.fd, self.initprocedure, snnopts)
            fc = MLP(self.fd, snnopts['mlpnodes'], snnopts['use_bn_mlp'])
            self.SBlocks.append(StructBlock(sb, fc, snnopts['use_bn_sblock']))

    
    def forward(self, input, bias=None):

        #input_shape = torch.tensor.size(input)
        #output = torch.tensor(self.nodes, input_shape[1], input_shape[2])

        inputshape = input.shape

        out = torch.empty(inputshape[0], self.SBlocks[0].SBlock.scaled_img_height, self.SBlocks[0].SBlock.scaled_img_width, self.numnodes).to('cuda')

        for i in range(self.numnodes):

            out[:, :, :, i] = self.SBlocks[i](input).squeeze() #Or try x.view(x.shape[0], x.shape[1], x.shape[2])

        out = out.permute(0, 3, 1, 2)

        return out


        #out = {}

        #for i in range(self.numnodes):

        #    out[i] = self.SBlocks[i](input)

        #return out


class ConvLayer(nn.Module):

    def __init__(self, inputchannels, convnodes, kernel, stride, padding, batchnorm=True, residual=False):
        super(ConvLayer, self).__init__()

        self.use_bn = False
        self.use_res = False
        self.bn = None

        self.conv = nn.Conv2d(inputchannels, convnodes, kernel, stride, padding)
        self.relu = nn.ReLU(inplace=False)


        if batchnorm == True:
            self.use_bn = True
            self.bn = nn.BatchNorm2d(convnodes)
        
        if residual == True and inputchannels == convnodes:
            self.use_res = True


    def forward(self, x):

        if self.use_res == True:
            identity = x

        out = self.conv(x)
        if self.use_bn == True:
            out = self.bn(out)
        if self.use_res == True:
            out += identity
        out = self.relu(out)

        return out




class SNN(nn.Module):


    def __init__(self, snnopts, inputchannels = 3, kernelsize = (3, 3)):

        super(SNN, self).__init__()

        self.imgh = snnopts['img_height']
        self.imgw = snnopts['img_width']
        self.fh = snnopts['field_height']
        self.fw = snnopts['field_width']
        self.sfr = snnopts['scale_factor_ratio']
        self.bsf = snnopts['base_scale_factor']
        self.use_bn_conv = snnopts['use_bn_conv']
        self.use_res_conv = snnopts['use_res_conv']
        self.tanhfactor = 1
        self.convnodes = snnopts['convnodes']
        self.numconvlayers = len(self.convnodes)
        self.spatialnodes = snnopts['spatialnodes']
        self.numspatiallayers = len(self.spatialnodes)
        self.convmods = torch.nn.ModuleList()
        self.smods = torch.nn.ModuleList()
        self.out_activation = nn.Tanh()

        #self.scale_factors = self.set_scale_factors(self.bsf)

        for idx in range(self.numconvlayers):

            if idx == 0:
                self.convmods.append(ConvLayer(inputchannels, self.convnodes[idx], kernelsize, stride = (1, 1), padding = (1, 1), batchnorm = self.use_bn_conv, residual = self.use_res_conv ))
            else:
                self.convmods.append(ConvLayer(self.convnodes[idx-1], self.convnodes[idx], kernelsize, stride = (1, 1), padding = (1, 1), batchnorm = self.use_bn_conv, residual = self.use_res_conv ))

        for idx in range(self.numspatiallayers):

            if idx == 0:
                self.smods.append(StructLayer(self.convnodes[-1], self.spatialnodes[idx], snnopts, stride = (1, 1)))
            else:
                self.smods.append(StructLayer(self.spatialnodes[idx-1], self.spatialnodes[idx], snnopts, stride = (1, 1)))


        self.scale_factors = self.set_scale_factors(self.bsf)


    def forward(self, input):

        x = input

        for idx in range(self.numconvlayers):

            x = self.convmods[idx](x)

        for idx in range(self.numspatiallayers):

            x = self.smods[idx](x)

        output = self.out_activation(x/self.tanhfactor)
        #output = torch.softmax(x, dim=1)
        #output = x

        return output


    def set_scale_factors(self, bsf):

        # Set smods = [sfr*sfr*bsf, sfr*bsf, bsf]. I.e. multiply each previous layer by the scale factor ratio
        self.bsf = bsf
        sfactors = []

        for s, smod in enumerate(self.smods):

            sf = ((self.sfr)**(self.numspatiallayers - s - 1))*bsf
            sfactors.append(sf)

            for sb in smod.SBlocks:

                sb.SBlock.set_transforms(sf, self.imgh, self.imgw) 

        return sfactors


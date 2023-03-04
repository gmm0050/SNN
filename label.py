import os.path as osp
import os
import argparse
import math
import random
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from bs4 import BeautifulSoup as BS

def gaussian2d(R, sigma, w):

    q = ( w*math.exp( -1*(R**2)/(2*sigma**2) ) ) # / (2*math.pi*(sigma**2)) (Use this denominator instead of one only if we want the sum of activations to equal zero, like a true probability distribution)

    return q

def add_kernel(Q, xloc, yloc, cl, is_trunc, is_diff, labelargs):


    #Read in arguments
    sigma = labelargs['sigma']
    weight = labelargs['weight']
    annformat = labelargs['format']
    wtruncdec = labelargs['weight_truncation_factor']
    sigmatruncinc = labelargs['sigma_truncation_factor']
    wdiffdec = labelargs['weight_difficulty_factor']
    sigmadiffinc = labelargs['sigma_difficulty_factor']

    shape = Q.shape
    rows = shape[1] #
    cols = shape[2]

    #Recalculate parameters based on whether the bounding box is truncated/difficult or not

    if is_trunc:
        
        weight = wtruncdec*weight
        sigma = sigmatruncinc*sigma

    if is_diff:

        weight = wdiffdec*weight
        sigma = sigmadiffinc*sigma

    kr = sigma*math.sqrt(2)*math.sqrt(math.log((255*weight)/0.5)) #Kernel Range of effect
    xrange = range(cols)[math.floor(xloc - kr):math.ceil(xloc + kr)]
    yrange = range(rows)[math.floor(yloc - kr):math.ceil(yloc + kr)]

    for x in xrange:

        for y in yrange:

            R = math.sqrt( (x - xloc)**2 + (y - yloc)**2 )

            g = gaussian2d(R, sigma, weight)

            Q[cl, y, x] += g ## Is this the right order for an image in pytorch tensor form?

    return Q


def label_target(target, center, labelargs):

    #Label Target images with probability distributions corresponding to the locations of recognizable objects in the image.
    #The center of each probability kernel should be located at the center of the bounding box, and have a predefined width based on
    #our confidence in its location. The height of the kernel should correspond to our GT confidence that there is an object at that location


    xc = center['xc']
    yc = center['yc']
    cl = center['cl']
    isdiff = center['isdiff']
    istrunc = center['istrunc']


    target = add_kernel(target, xc, yc, cl, istrunc, isdiff, labelargs)
    
    return target


def parse_args():

    parser = argparse.ArgumentParser(description='Label Training Images with Gaussian Probability Kernels.')

    parser.add_argument('datapath', type=str, help='Path to the Dataset Directory')
    parser.add_argument('savepath', type=str, help='Path to the Save Directory')
    parser.add_argument('-S','--sigma', type=float, default=1.0, help='Standard Deviation of Probability Kernel')
    parser.add_argument('-W','--kernel_weight', type=float, default=1.0, help='Max Height of Probability Kernel')
    parser.add_argument('-F','--format', type=str, default='XML', help='Annotation Format: JSON or XML')

    return parser.parse_args()

def choose_classes_random(datapath, numtochoose):

    #Read all Annotated Class folder names into a list

    classfolders = next(os.walk(datapath))[1]

    #classdict = {}

    #Choose a random subset of the imagenet classes

    subset = random.choices(classfolders, k=numtochoose)

    return subset


def make_classdict(subset):

    #Make classdict out of chosen subset of class folders

    num = len(subset)

    classdict = {}

    for i in range(num):

        classdict[subset[i]] = i

    return classdict


def make_label(target, centers, labelargs):

    #imagename is a six-digit number string associated with an imagenet image e.g. 010034, or 000127
    #classname is the name of the class e.g. n03273740, or n03388043
    #Datapath is the path to the root of the data
    #classdict is a dictionary that maps a classname to its index in the label / classification vector

    #Label Targets

    for center in centers:

        target = label_target(target, center, labelargs)


    #Return a dictionary with {classname, imagename, Label}

    #labeldict = {"classname" : classname,  "label" : target}

    return target



def batch_label(target, batchlabel, labelargs):

    #numbatches = target.shape[0]
    #Currently only works for one bounding box per image. We don't know how the label format will change when we load a 
    #dataset with multiple bounding boxes per image

    batchsize = target.shape[0]
    #target = torch.zeros(targetshape)

    for b in range(batchsize):

        tgt = torch.zeros( target.shape[1:] )

        if batchlabel['cl'][b] == None:
            pass

        else:
            xc = batchlabel['xc'][b].float().item()
            yc = batchlabel['yc'][b].float().item()
            cl = batchlabel['cl'][b].item()
            isdiff = batchlabel['isdiff'][b].item()
            istrunc = batchlabel['istrunc'][b].item()
            #print("b is:")
            #print(b)
            #print("")
            target[b, :, :, :] = add_kernel(tgt, xc, yc, cl, isdiff, istrunc, labelargs)
    
    if labelargs['use_nullclass'] == True:
        nulltensor = torch.ones( [batchsize, target.shape[2], target.shape[3]] ) - torch.sum(target, dim=1)
        #print("This is the size of the null tensor:")
        #print(nulltensor.shape)
        #print("")
        #print("This is the size of the target:")
        #print(target.shape)
        #print("")
        target = torch.cat( (target, nulltensor.unsqueeze(dim=1)), dim=1 )

    return target


def batch_segmentation_label(targetshape, seglabel, labelargs):

    ## Generates the Target Tensor given a segmentation label, in which each pixel has a value that corresponds to the id for 
    ## a particular class. By default a probability of 1 is given to each pixel in the segmented region.

    batchsize = targetshape[0]
    target = torch.zeros(targetshape)


    for b in range(batchsize):

        tgt = torch.zeros( targetshape[1:] )

        for i, id in enumerate(labelargs["label_ids"]):

            tgt[i, :, :] = (seglabel[b, :, :] == id).float()

        target[b, :, :, :] = tgt

    return target



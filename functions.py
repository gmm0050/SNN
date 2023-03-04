
## --- Functions for Use in NN Architectures or in General ---

import torch
import math
#import torchvision.transforms as T


def FieldMap(xindices, yindices):

    field = torch.meshgrid(xindices, yindices, indexing = 'ij')

    return field[0], field[1]


def SpatialVariance(P, fh, fw):


    mh = math.floor(fh/2)
    mw = math.floor(fw/2)

    x = torch.linspace(-mw, mw, steps = fw)
    y = torch.linspace(-mh, mh, steps = fh)

    fieldx, fieldy = FieldMap(x, y)

    field = torch.stack((fieldx, fieldy), dim = 0)

    field = torch.flatten(field, start_dim = 1, end_dim = -1)

    field = field.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    field = field.to('cuda:0')

    # Uncomment line below to run all below lines in a single line
    # Var = torch.sum(P*torch.sum(torch.square(field - torch.sum(P.unsqueeze(2)*field, dim = 3).unsqueeze(3)), dim = 2), dim = 2)

    P = P.unsqueeze(2)

    mu = torch.sum(P*field, dim = 3)

    mu = mu.unsqueeze(3)

    Fsq = torch.square(field - mu)

    Dsq = torch.sum(Fsq, dim = 2)

    P = P.squeeze(2)

    Var = torch.sum(P*Dsq, dim = 2)

    return Var


def InvVarWeighting(P, fh, fw, k1, k2):

    # Uncomment line below to run all below lines in a single line
    # weight = (k1 / (SpatialVariance(P, fh, fw) + k2)).unsqueeze(2)

    Var = SpatialVariance(P, fh, fw)

    weight = k1 / (Var + k2)

    weight = weight.unsqueeze(2)

    return weight



def calc_iou(outputs: torch.Tensor, labels: torch.Tensor):
    eps = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

def iou_loss(outputs, labels, threshold):

    outputs = (outputs >= threshold)

    iou = calc_iou(outputs, labels)

    return iou
    
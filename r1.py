import os.path as osp

import numpy as np
#from numpy.core.defchararray import _binary_op_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F



#R1 Reward Function for the Spatial-Structure Neural Network (SSNN)


#Set hyperparameters alpha and beta in the args

class Similarity_Reward(nn.Module):

    def __init__(self, opts):

        super(Similarity_Reward, self).__init__()

        self.alpha = opts['alpha'] #Overall Reward Multiplier
        self.beta = opts['beta'] #True Negative to True Positive reward ratio 

    def forward(self, output, target):

        #The Similarity Reward is maximized when P = Q, e.g. when the predicted probability is equal to the ground truth probability
        #It is Probabilistically ill-defined, but appears to be related to the probability of a match between ground truth and a random sample with probability P
        # SR = P*Q + (1 - P)*(1 - Q + P) or alternatively (1 - (P - Q))*(1 - (Q - P))
        #We can adjust the reward function with a positive definite multiplier A(Q), which acts as an attention/weighting mechanism
        #The purpose of this adjustment is to reduce the magnitude of the gradient for particular cases where we expect to learn very little

        O = torch.ones(output.shape).to('cuda')

        #Calculating Attention Parabola Parameters
        a = 2*self.alpha*self.beta
        b = self.alpha*(1 - 3*self.beta)
        c = self.alpha*self.beta

        #Creating the Attention Tensor
        A = a*(target**2) + b*(target) + c

        rewards = A * ( output*target + (O - output)*(O - target + output) )

        #SR = torch.sum(rewards)

        SL = -1*torch.sum(rewards)

        return SL

    #def backward(self, P, Q):

    #    invgrad = 2*(P - Q)

    #    return invgrad

        

#R1 Reward function

class R1_Reward(nn.Module):

    def __init__(self, opts):

        super(R1_Reward, self).__init__()

        self.alpha = opts['alpha']
        self.beta = opts['beta']


    def forward(self, output, target):

        # P = output / Q = target
        # R1 = max( alpha*P*Q , beta*(1-P)*(1-Q) ) , e.g. max( TP Reward, TN Reward )

        #Assert output.shape = target.shape = 1-dim

        #Create a vector of all ones to form the contrapositives (?)
        O = torch.ones(output.shape)

        rewards = torch.maximum( self.alpha*torch.dot(output, target), self.beta*torch.dot(O - output, O - target) ) #Definition of the R1 Reward Function: R1 = max[ alpha*P*Q, beta*(1-P)*(1-Q) ]

        R1 = torch.sum(rewards) #We sum rewards across all of the used pixels

        return R1

    #def backward(self, ):
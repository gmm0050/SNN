#Training Script for the Spatial Structure Neural Network (SSNN)

#

import os.path as osp
import os
import math
import time

from PIL import Image

import numpy as np
from numpy.core.defchararray import _binary_op_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import nevergrad as ng

import label
import snnutils



# Split into Training / Validation / Testing Subsets



# Shuffle Training samples, so that batches are composed of mixed/varied classes, using the Pytorch Dataloader

#loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, batch_sampler=True, )

# Train on Training Subset

class coach():


    def __init__(self, net, optimizer, objective_function, labelargs, traindata, valdata, testdata, train_loadargs, val_loadargs, test_loadargs, opts):

        self.net = net
        self.opt = optimizer
        self.lf = objective_function
        self.labelargs = labelargs
        self.base_scales = opts['base_scale_factors'] ##Where do we set the scales to search over?

        self.device = 'cuda:0'
        self.global_step = 0
        self.max_step = opts['max_step']
        self.test_step = 0

        self.traindata = traindata
        self.valdata = valdata
        self.testdata = testdata

        self.train_loadargs = train_loadargs
        self.val_loadargs = val_loadargs
        self.test_loadargs = test_loadargs

        self.traintargetshape = (train_loadargs["batchsize"], train_loadargs["numclasses"], train_loadargs["size"][0], train_loadargs["size"][1])
        self.valtargetshape = (val_loadargs["batchsize"], val_loadargs["numclasses"], val_loadargs["size"][0], val_loadargs["size"][1])
        self.testtargetshape = (test_loadargs["batchsize"], test_loadargs["numclasses"], test_loadargs["size"][0], test_loadargs["size"][1])

        self.trainobjectives = []
        self.valobjectives = []
        self.testobjectives = []

        self.testprobabilities = []
        self.testouts = []
        self.testclasslabels = []
        self.testtargets = []
        self.testcounter = 0

        self.val_interval = opts['val_interval']
        self.best_val_loss = None

        self.plot_test_error = opts['plot_test_error']
        self.numtest = opts['numtest']
        self.test_interval = opts['test_interval']

        self.epochs = []

        self.activations = {}
        self.requested_nodes = []

        #Operations
        self.set_ops(self.train_loadargs["resize"][0], self.train_loadargs["resize"][1])

    def set_ops(self, imgheight, imgwidth):

        Hout = math.floor(imgheight / self.net.bsf)
        Wout = math.floor(imgwidth / self.net.bsf)
        self.downsample_tgt = T.Resize((Hout, Wout))
        self.downsample_input = T.Resize((imgheight, imgwidth))

        return True
        

    def train(self, totalsteps):
        
        self.net.to('cuda')
        self.net.train()
        self.max_step = totalsteps
        self.trainobjectives = torch.empty(len(self.base_scales), self.max_step)
        self.set_ops(self.train_loadargs["resize"][0], self.train_loadargs["resize"][1])
        #trainobjectives = []
        #trainouts = []

        

        while self.global_step < self.max_step:

            #Display current iteration every so often to gage where we are in the training process
            if self.global_step % 10 == 0:
                print("At Iteration {}".format(self.global_step))
                print("")
            
            #Read next input/label pair from collected training data
            data = next(iter(self.traindata))
            if self.labelargs['segmentation'] == False:
                x = data['input']
                l = data['label']
            elif self.labelargs['segmentation'] == True:
                x = data[0]
                l = data[1]

            if type(l) is list:
                batchlabel = l[0]
            elif type(l) is dict:
                batchlabel = l
            else:
                batchlabel = l

            #Generate the target output for the data batch
            target = torch.zeros(self.traintargetshape)
            target = self.batch_label(target, batchlabel, self.labelargs)

            #Test adding a small constant across the image
            if self.labelargs['use_nullclass'] == False:
                target = target + 0.015
                target = torch.clamp(target, min = 0.0, max = 1.0)

            #Downsample to get a target that is the same size as the output
            target = self.downsample_tgt(target)
            x = self.downsample_input(x)
            
            #Forward and Backward Pass over a set of scales / windows / focal lengths
            for s, scale in enumerate(self.base_scales):
                    
                #self.net.set_scale_factors(scale) #This only works if we are using an SNN. Can we move this outside of coach definition?

                out, target = self.forwardpass(x, target)

                #trainouts[batch_idx] = out

                trainobj = self.calc_objective(out, target)

                self.trainobjectives[s, self.global_step] = trainobj

                self.opt.zero_grad()

                trainobj.float().backward()

                self.opt.step()


            # Validation 

            if self.global_step % self.val_interval == 0 or self.global_step == self.max_step:

                val_loss_dict = self.validate()

                if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):

                    self.best_val_loss = val_loss_dict['loss']


            # Intermediate Testing

            if (self.global_step % self.test_interval == 0 or self.global_step == self.max_step) and self.plot_test_error == True:

                self.test(self.numtest)

                self.testcounter += 1

                # Indicate when Training is done


            # Update Step Counter

            self.global_step += 1


            if self.global_step == self.max_step:

                print("Training Finished.")

                break
                

        return self.trainobjectives #, trainouts




    def validate(self):

        self.net.eval()
        val_loss_dict = {}
        self.set_ops(self.val_loadargs["resize"][0], self.val_loadargs["resize"][1])


        valdata = next(iter(self.valdata))

        if self.labelargs['segmentation'] == False:
            x = valdata['input']
            l = valdata['label']
        elif self.labelargs['segmentation'] == True:
            x = valdata[0]
            l = valdata[1]

        if type(l) is list:
            batchlabel = l[0]
        elif type(l) is dict:
            batchlabel = l
        else:
            batchlabel = l
        valtarget = torch.zeros(self.valtargetshape)
        valtarget = self.batch_label(valtarget, batchlabel, self.labelargs)

        if self.labelargs['use_nullclass'] == False:
            valtarget = valtarget + 0.015
            valtarget = torch.clamp(valtarget, min = 0.0, max = 1.0)
        #valtarget = valtarget + 0.015

        valtarget = self.downsample_tgt(valtarget)
        x = self.downsample_input(x)

        valobj = []

        with torch.no_grad():

            for s, scale in enumerate(self.base_scales):
                    
                #self.net.set_scale_factors(scale)

                out, valtarget = self.forwardpass(x, valtarget)

                valobj.append(self.calc_objective(out, valtarget))

        self.valobjectives.append(max(valobj).item())

        val_loss_dict['loss'] = max(valobj)
        val_loss_dict['flag'] = False ##Write validation criterion into here

        self.net.train()

        return val_loss_dict



    def test(self, test_iterations):

        self.net.eval()
        self.set_ops(self.test_loadargs["resize"][0], self.test_loadargs["resize"][1])

        #testouts = []
        #testrewards = []
        #testprobabilities = []
        print("")
        print("Testing Number {}!".format(str(self.testcounter)))
        print("")
        self.testclasslabels.append([])
        self.testobjectives.append([])
        self.testprobabilities.append([])
        self.testouts.append([])
        self.testtargets.append([])
        print("")
        print("Length of testouts:")
        print(len(self.testouts))
        print("")
        time.sleep(5)
        self.test_step = 0
        while self.test_step < test_iterations:

            testdata = next(iter(self.testdata))

            if self.labelargs['segmentation'] == False:
                x = testdata['input']
                l = testdata['label']
            elif self.labelargs['segmentation'] == True:
                x = testdata[0]
                l = testdata[1]
            
            if type(l) is list:
                batchlabel = l[0]
            elif type(l) is dict:
                batchlabel = l
            else:
                batchlabel = l

            target = torch.zeros(self.testtargetshape)
            target = self.batch_label(target, batchlabel, self.labelargs)

            if self.labelargs['use_nullclass'] == False:
                target = target + 0.015
                target = torch.clamp(target, min = 0.0, max = 1.0)
            #target = target + 0.015

            target = self.downsample_tgt(target)
            x = self.downsample_input(x)

            testobjs = []

            with torch.no_grad():
                
                for s, scale in enumerate(self.base_scales):
                    
                    #self.net.set_scale_factors(scale)
                    
                    out, target = self.forwardpass(x, target)

                    testobjs.append(self.calc_objective(out, target))

                    outs = snnutils.split_batch_to_list(out, dim=0)

                    probabilities = self.classify(outs)


                self.testobjectives[-1].append(max(testobjs))

                self.testprobabilities[-1].append(probabilities)

            self.testouts[-1].append(outs)
            self.testtargets[-1].append(target)
            if self.labelargs['segmentation'] == False:
                classlabel = batchlabel['cl']
                self.testclasslabels[-1].append(classlabel.item())

            self.test_step += 1

        print("Length of Testouts after Testing:")
        print(len(self.testouts[-1]))
        print("")
            
        return self.testobjectives, self.testprobabilities #, testouts

    
    def forwardpass(self, x, y):
        
        x, y = x.to(self.device).float(), y.to(self.device).float()

        out = self.net.forward(x)

        return out, y

    def calc_objective(self, out, y):

        objective = self.lf(out, y)

        return objective


    def classify(self, outlist):

        classprobs = []

        for idx, outtensor in enumerate(outlist):

            clp = []
            
            for i in range(outtensor.shape[1]):

                clp.append(torch.max(outtensor[:, i, :, :]).item())

            
            classprobs.append(clp)

        return classprobs

    def get_module_names(self):

        modnames = []

        for name, module in self.net.named_modules():

            modnames.append(name)

        self.modulenames = modnames

        return modnames

    def get_activation(self, name):

        def hook_fn(m, i, o):

            self.activations[name] = o.detach()

        return hook_fn

    

    def request_node_output(self, nodes):

        self.requested_nodes.extend(nodes)
        
        exitflag = True

        for dict in nodes:

            if dict['type'] == 'conv':

                mod = self.net.convmods[dict['layer']]
                mod.register_forward_hook(self.get_activation(dict['name']))

            elif dict['type'] == 'sblock':

                mod = self.net.smods[dict['layer']].SBlocks[dict['node']].SBlock
                mod.register_forward_hook(self.get_activation(dict['name']))

            else:

                print("The node type must be either:")
                print(['conv', 'sblock'])
                exitflag = False
            
        return exitflag


    def set_base_scales(self, scale_factors):

        self.base_scales = scale_factors

        return True

    
    def batch_label(self, shape, batchlabel, args):

        if args['segmentation'] == True:

            tgt = label.batch_segmentation_label(shape, batchlabel, args)

        else:

            tgt = label.batch_label(shape, batchlabel, args)

        return tgt




            


    

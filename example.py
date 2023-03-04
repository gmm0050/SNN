# An Example Script for Training and Testing the SNN on a dataset

# Import Libraries, utilities, etc...

import os
import os.path as osp
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import snnutils
import label
import visualize as viz
import sblock
import r1
import coach

import plotly.graph_objects as go
import matplotlib.pyplot as plt


# (0) Set Hyperparameters

#Set Total Training Iterations and validation interval

total_iterations = 100
val_interval = 10
threshold = 0.8

#Set Dataset to Train and Test on 

dataset = 'imagenet' #Name of the dataset
datapath = '../../../../media/beefysim/lacie/imagenet/' #Path to root of Dataset
traindatapath = ""
valdatapath = ""
testdatapath = ""
annotationpath = ""
annotationformat = 'xml'

#Splits are used exclusively for the Pascal Dataset
trainsplits = None
valsplits = None
testsplits = None


#Set Batch Size for Training / Testing / Validation

train_batch_size = 16 #Number of Image Tensors to train on at one time
val_batch_size = 4 #Number of Image Tensors to validate on at one time
test_batch_size = 1 #Number of Image Tensors to test on at one time

#Set Perceptual Field Characteristics

hf = 7 #Perceptual Field height
wf = 7 #Perceptual Field width
scale_factor = 0.5 #Scale Factor from Perceptual Field in the ith layer and the i-1st layer

#Set Number of Layers and Number of Nodes in each Layer

numconvlayers = 3 #Number of Convolutional layers in the Network Stem
numspatiallayers = 4 #Number of spatial layers in the Network including the final layer

convnodes = [] #Number of nodes in each convolutional layer
spatialnodes = [] #Number of nodes in each spatial layer
mlpnodes = [] #Number of nodes in each Sblock's mlp

#Choose whether to use additional NN methods/techniques/components
use_mlp = True #Boolean whether to add an mlp after each spatial block
use_dropout = False #Boolean whether to use dropout or not
use_residual = False #Boolean whether to use ResNet-like residual features #(Not Implemented)
use_data_augmentation = False #Boolean whether to augment dataset with affinely-transformed examples #(Not Implemented)
use_transfer_learning = False #Boolean whether to train one layer at a time #(Not Implemented)

#Choose Optimization and Initialization Procedures

init_procedure = 'xavier_uniform' #There are 4 standard pytorch weight initialization procedures: (1) xavier uniform , (2) xavier normal, (3) kaiming uniform, (4) kaiming normal
opt_procedure = 'AdaDelta' #Optimizer Name

#Set NN Hypers
eps = 0.000001
learning_rate = 0.1
momentum = 0.9
regularization = 0.01
decay_rate = 0.0001

base_scales = [4]
base_scale_factor = base_scales[0]
scale_factor_ratio = 0.6

normalization_mean = [0.485, 0.456, 0.406] #Normalize Images around the mean of the dataset
normalization_std = [0.229, 0.224, 0.225] #Normalize standard deviation of images using the std of the dataset

widthtrain = 256
heighttrain = 256

widthval = 256
heightval = 256

widthtest = 256
heighttest = 256

#Set R1 parameters

alpha_r1 = 1 #True Positive Reward Function Weight
beta_r1 = 0.04 #True Negative Reward Function Weight

#Set Label parameters

sigma_kernel = 1 #Kernel Width Parameter
weight_kernel = 1 #Kernel Max Height / Probability

wtruncdec = 0.8
sigmatruncinc = 2

wdiffdec = 0.7 #Scaling Factor on Kernel Height when bounding box is labelled "difficult"
sigmadiffinc = 1.5 #Scaling Factor on Kernel Width when bounding box is labelled "difficult"

#Set Visualization Options

viz_label = True #Visualize a Label as a grayscale image 
viz_activation = True #Visualize a node activation
viz_field = True #Visualize a Structure Block perceptual field with a heat map
viz_deepdream = True #Visualize the Activation pattern of a node using a deep dream
viz_superimposed = True #Visualize an Image with a Transparent Probability Distribution superimposed

activation_nodenames = ['convmods.2', 'smods.0.SBlocks.7.SBlock']
activation_node_inputs = 

deepdream_nodenames = []
deepdreamnode_inputs = 

nodenames = activation_nodenames.extend(deepdream_nodenames)

# (1) Initialization and Preprocessing


#Set MLP / SNN / Training / Optimizer opts



#Arguments for the SNN Architecture 
snnopts = {}
snnopts['init_procedure'] = init_procedure
snnopts['convnodes'] = convnodes
snnopts['spatialnodes'] = spatialnodes
snnopts['mlpnodes'] = mlpnodes
snnopts['field_height'] = hf
snnopts['field_width'] = wf
snnopts['scale_factor_ratio'] = scale_factor_ratio
snnopts['base_scale_factor'] = base_scale_factor
snnopts['use_mlp'] = use_mlp
snnopts['use_dropout'] = use_dropout
snnopts['use_residual'] = use_residual
snnopts['use_augmentation'] = use_data_augmentation
snnopts['use_transfer_learning'] = use_transfer_learning


#Arguments for Training and Validation Loops
trainopts = {}
trainopts['max_step'] = total_iterations
trainopts['val_interval'] = val_interval


#Arguments for the Optimizer
optopts = {}
optopts['learning_rate'] = learning_rate
optopts['momentum'] = momentum
optopts['reg'] = regularization
optopts['decay'] = decay_rate



#Arguments for the Reward Function
lfopts = {}
lfopts['alpha'] = alpha_r1
lfopts['beta'] = beta_r1


#Arguments for Making Target Labels
label_args = {}
label_args['sigma'] = sigma_kernel
label_args['weight'] = weight_kernel
label_args['format'] = annotationformat
label_args['weight_truncation_factor'] = wtruncdec
label_args['sigma_truncation_factor'] = sigmatruncinc
label_args['weight_difficulty_factor'] = wdiffdec
label_args['sigma_difficulty_factor'] = sigmadiffinc

# Arguments for Loading Datasets
train_load_args = {}
train_load_args["datapath"] = datapath
train_load_args["annpath"] = annotationpath
train_load_args["imgpath"] = traindatapath
train_load_args["splits"] = trainsplits
train_load_args["resize"] = (widthtrain, heighttrain)
train_load_args["batchsize"] = train_batch_size

val_load_args = {}
val_load_args["datapath"] = datapath
val_load_args["annpath"] = annotationpath
val_load_args["imgpath"] = valdatapath
val_load_args["splits"] = valsplits
val_load_args["resize"] = (widthval, heightval)
val_load_args["batchsize"] = val_batch_size

test_load_args = {}
test_load_args["datapath"] = datapath
test_load_args["annpath"] = annotationpath
test_load_args["imgpath"] = testdatapath
test_load_args["splits"] = testsplits
test_load_args["resize"] = (widthtest, heighttest)
test_load_args["batchsize"] = test_batch_size

#Create Preprocess Image Transforms
image_transforms = T.compose([ T.Resize() , T.CenterCrop() , T.ToTensor() , T.Normalize( mean = normalization_mean, std = normalization_std) ])

#Load in Training and Validation Datasets

traindataset, trainobjlabels, trainclasslabels, classdict = snnutils.load_dataset(dataset, train_load_args, image_transforms, label_args)

valdataset, valobjlabels, valclasslabels = snnutils.load_dataset(dataset, val_load_args, image_transforms, label_args)

testdataset, testobjlabels, testclasslabels = snnutils.load_dataset(dataset, test_load_args, image_transforms, label_args)


TrainData = torch.utils.data.DataLoader(traindataset, batch_size=train_batch_size, shuffle=True, batch_sampler=True, )

ValData = torch.utils.data.DataLoader(valdataset, batch_size=val_batch_size, shuffle=True, batch_sampler=True, )

TestData = torch.utils.data.DataLoader(testdataset, batch_size=test_batch_size, shuffle=True, batch_sampler=True, )

nc = len(classdict)

train_load_args["numclasses"] = nc
val_load_args["numclasses"] = nc
test_load_args["numclasses"] = nc

# (2) Training

net = sblock.SNN(snnopts)

params = net.parameters()
optimizer = torch.optim.Adadelta(params, rho = optopts['momentum'], eps = optopts['eps'], lr = optopts['learning_rate'], weight_decay = optopts['decay'])
objective_function = r1.Similarity_Reward(lfopts)

snncoach = coach.coach(net, optimizer, objective_function, label_args, TrainData, ValData, TestData, train_load_args, val_load_args, test_load_args, trainopts)

snncoach.set_base_scales(base_scales)

all_module_names = snncoach.get_module_names()

snncoach.request_node_output(nodenames)

trainobjs = snncoach.train(total_iterations)


# (3) Testing

testobjs, testprobabilities = snncoach.test(10)


# (4) Plots and Statistical Comparisons to Baseline

# Calculate Statistics


acc, tpr, fpr, prec, recall, f1, tp, fp, fn, tn = snnutils.classify_with_threshold(testprobabilities, testobjlabels, threshold)

acctop1 = snnutils.classify_topn(testprobabilities, testclasslabels, 1)

acctop5 = snnutils.classify_topn(testprobabilities, testclasslabels, 5)


# Generate ROC curve for SNN

roc = {}
roc['x'] = []
roc['y'] = []
rocs = []
numthresholds = 100
thresholds = np.linspace(0, 1, numthresholds)

for ts in thresholds:

    curracc, currtpr, currfpr = snnutils.classify_with_threshold(testprobabilities, testobjlabels, ts)

    roc['x'].append(currfpr)
    roc['y'].append(currtpr)

rocs.append(roc)

#Append other baseline rocs (AlexNet / Resnet50 / InceptionV4 / ViT)

# Visualizations

batch_idx = 3
idx_in_batch = 10
cl = TestData[batch_idx]['class'] #Display the target pdf for the class that the image is labelled with
label_tensor = TestData[batch_idx]['target'][idx_in_batch, :, :, cl] #Extract the particular target pdf to display as a grayscale image
target_map = viz.vis_label(label_tensor, cl, want_show=viz_label)


heatmapnode = 
heatmapfeature = 
colormin =
colormax = 
field_heatmap = viz.vis_field_heatmap(net, heatmapnode, heatmapfeature, colormin, colormax, want_show=viz_field)


actmapnode = 
actmapinput = 
activation_map = viz.vis_node_activation(net, actmapnode, actmapinput, want_show=viz_activation)

ddnode = 
ddinput = 
deep_dream = viz.vis_deep_dream(net, ddnode, ddinput, want_show=viz_deepdream)

img_su = 
tgt_su = 
color_su = np.array([3, 190, 252])
img_target_superimposed = viz.vis_target_superimposed(img_su, tgt_su, color_su, maxopacity_su = 0.80, want_show=viz_superimposed)



# Compare Classification Accuracy to Baselines : YOLOv3 / Imagenet / COCO / MNIST / REAL / ObjectNet

# Table of Accuracies on different datasets vs best + baselines

result_table = viz.make_table()

# Plot ROC Curve vs baselines

roc_plot = viz.plot_roc(rocs, thresholds)

# 

# Training Error vs Iteration + Validation Error vs Iteration (Compare to Baselines)

trainerrors = {}
trainerrors['x'] = snncoach.epochs
trainerrors['y'] = snncoach.trainobjectives
trainerrors['name'] = 'training reward'
trainerrors['color'] = 'firebrick'
trainerrors['width'] = 3
trainerrors['linemode'] = 'lines+markers'

valerrors = {}
valerrors['x'] = snncoach.epochs
valerrors['y'] = snncoach.valobjectives
valerrors['name'] = 'validation reward'
valerrors['color'] = 'royalblue'
valerrors['width'] = 3
valerrors['linemode'] = 'lines+markers'

errordata = []
errordata.append(trainerrors)
errordata.append(valerrors)

errorplotlabels = {}
errorplotlabels['title'] = 'Training and Validation Reward vs Training Iteration'
errorplotlabels['xaxis'] = 'Training epoch'
errorplotlabels['yaxis'] = 'Reward'

val_and_train_error_plot = viz.plot_errors(errordata, errorplotlabels)

# Plot Reward Surface in 3D Plot

reward_function_plot = viz.plot_reward_surf(25, 25)

# Plot only Validation Errors vs Training Iteration

valerrordata = []
valerrordata.append(valerrors)

valerror_plot = viz.plot_errors(valerrordata, errorplotlabels)


## Compare SNN Performance on Adversarial Images (Corrupted/Perturbed/Abstract Patterns/Noise) to CNN Performance 

## Compare Classification Performance of SNN Variants


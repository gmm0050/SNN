#Visulization script for the Spatial Structure Neural Network

#import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import label
#import r1
import snnutils

import plotly.graph_objects as go
import plotly.express as px


#Visualize labels


#Visualize as 1-channel (gray-scale) images

def vis_label(label_tensor, cl, want_show=True):


    label_np = label_tensor[cl,:,:].cpu().numpy() # Convert tensor to Numpy Array
    label_np = snnutils.rescale_label_np(label_np, 0, 1) #Rescale Label Tensor values (0-1) to Intesity values
    label_im = snnutils.np2im(label_np) # Convert Numpy Array to PIL image
    label_im = ImageOps.grayscale(label_im)

    if want_show == True:

        label_im.show() # Show Grayscale Label image

    return label_im, label_np
    


#Visualize Hidden Layers


def vis_node_activation(activationmap, grayscale = True, want_show=True):

    actmap_np = activationmap.cpu().numpy()

    actmap_np = (actmap_np * 255).astype(np.uint8)

    actmap = snnutils.np2im(actmap_np)

    if grayscale == True:
        actmap = ImageOps.grayscale(actmap)

    if want_show == True:

        actmap.show()

    return actmap



#Visual Field Heat Map (Plot SBlock weights for one feature as a Heat Map)

    #layer = 2 #Layer we want to extract the Visual field weights from
    #f = 0 #SBlock we want to extract the weights from 
    #weight_map = net.smod[layer].S[f].weight
    #weight_map = weight_map.cpu().numpy()
    #wmap = cv2.imwrite(weight_map, "weight_map.png")
    #cv2.applyColorMap(wmap, cv2.COLORMAP_JET)
    #cv2.imshow(imcounter, wmap)
    #imcounter += 1



def vis_field_heatmap(net, node, feature, vmin, vmax, want_show=True, colormap='jet'):

    # net is the SNN
    # node is the output node for the Structure Block
    # feature is the input node for which we take a slice of the Structure Block and inspect it

    # Extract weights from net

    field = net.StructLayer(node.layer).SBlock(node.number)[:, :, feature] ## Look at Pytorch SNN class structure to correct this

    # Convert to Numpy Array and Rescale to (0 - 255)

    field_np = field.cpu().numpy()

    # Apply Colormap to Numpy Array using matplotlib colormap

    plt.figure()
    plt.pcolor(field_np, cmap = colormap, vmin=vmin, vmax = vmax)
    plt.title("Field Heatmap")

    # Convert to PIL image

    field_im = snnutils.np2im(field_np)

    # Show PIL image

    if want_show == True:

        plt.show()

    return field_im




    # Deep Dream / Saliency Map

def objective_L2(dst):

    dst.diff[:] = dst.data



def dreamstep(net, node, img, activationmap, iteration, opts, objective=objective_L2):

    # Perform an update that transforms the input such that the activation of the given node is closer to the desired activation map
    ## Alternative to using gradient ascent to perform the optimization, we could potentially use CMA

    # (0) Get node activation from input img using net
    out = net(img)
    activation = out[node['layer']][node['number']]

    # (1) Calculate Reward
    obj = objective(activation, activationmap)

    obj.backward()

    # (2) Calculate gradient
    grad = img.grad.data

    if opts['smoothing'] == True:

        # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
        # sigma is calculated using an arbitrary heuristic feel free to experiment
        sigma = ((iteration + 1) / opts['num_gradient_ascent_iterations']) * 2.0 + opts['smoothing_coefficient']
        grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    if opts['normalization'] == True:

        # Normalize the gradients (make them have mean = 0 and std = 1)
        # I didn't notice any big difference normalizing the mean as well - feel free to experiment
        g_std = torch.std(grad)
        g_mean = torch.mean(grad)
        grad = grad - g_mean
        grad = grad / g_std

    # (3) Update Image with gradient
    img.data += opts['lr'] * grad 

    # (4) Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    img.grad.data.zero_()
    ub = 255
    lb = 0
    img.data = torch.max(torch.min(img, ub), lb)


    


def vis_deep_dream(net, node, img, activationmap, preprocess, objfun, opts, want_show=True):

    # Generate a Deep Dream from the input by optimizing the input to approximate a specific activation pattern for the chosen node
    # The octave base is the input image minus the mean of the network
    #  

    img = preprocess(img)
    base_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(opts['pyramid_size']):

        new_shape = utils.get_new_shape(opts, base_shape, pyramid_level)
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)

        for iteration in range(opts['num_gradient_ascent_iterations']):

            
            if opts['jitter'] == True:

                h_shift, w_shift = np.random.randint(-opts['spatial_shift_size'], opts['spatial_shift_size'] + 1, 2)
            
                input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)

                dreamstep(net, node, input_tensor,activationmap, iteration, opts, objective=objfun)

                input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)

            else:

                dreamstep(net, node, input_tensor,activationmap, iteration, opts, objective=objfun)

    outputimgnp = input_tensor.cpu().numpy()

    outputimg = snnutils.np2im(outputimgnp)

    if want_show == True:

        outputimg.show()

    return outputimg




def vis_target_superimposed(imagetensor, target_tensor, cl, color, maxopacity = 0.80, want_show=True):

    trans = T.ToPILImage()

    #Convert Tensors to Numpy Arrays
    target_np = target_tensor[:, :, cl].cpu().numpy()

    image_np = imagetensor.cpu().numpy()

    if image_np.shape[0] == 1: #Convert 1-Channel Grayscale Image to 3-Channel Image
        image_np = np.squeeze(image_np)
        image_np = np.stack( (image_np,)*3, axis=0)
        image_torch = torch.from_numpy(image_np)
        imgray = trans(image_torch)
        imgray.show()

    #Generate the Target and Image Opacity arrays based on the Target value at a pixel
    alpha = target_np*maxopacity

    contralpha = np.ones(alpha.shape) - alpha

    #Create a numpy array with the color we chose for the Target's kernels
    kernel_color_np = np.ones(image_np.shape)*color[:, np.newaxis, np.newaxis]


    #Add together the image and the semi-transparent target
    im = alpha[np.newaxis, :, :]*kernel_color_np + contralpha[np.newaxis, :, :]*image_np

    #Convert numpy array to PIL image
    imtensor = torch.from_numpy(im)
    output_im = trans(imtensor)

    #Display and return image

    if want_show == True:

        output_im.show()

    return output_im



def plot_errors(data, labels):

    #Plot the set of errors as the y-axis against the corresponding iterations on the x-axis

    #The x-data and y-data are stored in a list of dictionaries with template: {'x': x, 'y': y, 'name': name}
   
    fig = go.Figure()

    for series in data:

        xvals = series['x']
        yvals = series['y']
        name = series['name']
        color = series['color']
        width = series['width']
        linemode = series['linemode']

        fig.add_trace(go.Scatter(name = name, x = xvals, y = yvals, mode=linemode, line = dict(color = color, width = width)))
    
    title = labels['title']
    xaxis = labels['xaxis']
    yaxis = labels['yaxis']

    fig.update_layout(title = title, xaxis_title = xaxis, yaxis_title = yaxis)

    return fig


    
def plot_roc(rocs, labels):

    #Plot the set of roc curves (X: False Positive Rate , Y: True Positive Rate)

    fig = go.Figure()

    for roc in rocs:

        name = roc['name']
        xvals = roc['x']
        yvals = roc['y']
        color = roc['color']
        width = roc['width']
        linemode = roc['linemode']

        fig.add_trace(go.Scatter(name = name, x = xvals, y = yvals, mode = linemode, line = dict(color = color, width = width) ) )

    title = labels['title']
    xaxis = labels['xaxis']
    yaxis = labels['yaxis']

    fig.update_layout(title = title, xaxis_title = xaxis, yaxis_title = yaxis)

    return fig



def make_table(rowtitle, rownames, colnames, valarray):

    #Make a Table Comparing the Comparison of Our SNN with a myriad of baselines on a set of Image Datasets / Battery of Tests

    assert valarray.shape[0] == len(rownames)
    assert valarray.shape[1] == len(colnames)

    headervals = [rowtitle]
    headervals.append(colnames)

    cellvals = [ rownames ]

    for colidx in range(len(valarray.shape[1])):

        column = valarray[:, colidx]
        cellvals.append(column.tolist())

    
    table = go.Figure(data = go.Table( header = dict(values = headervals, line_color = 'darkslategray', fill_color = 'white', font = dict(color = 'black', size = len(headervals))), cells = dict(values = cellvals, line_color = 'darkslategray', fill_color = 'white', font = dict(color = 'black', size = len(cellvals))) ))

    return table



def plot_reward_surf(numxvals, numyvals, opts):

    #Plot the Reward Surface as a 3D Heatmap

    alpha = opts['alpha']
    beta = opts['beta']
    colorscales = px.colors.named_colorscales()

    xvals = np.linspace(0, 1, numxvals)
    xv = np.reshape(xvals, (1, xvals.size))
    yvals = np.linspace(0, 1, numyvals)
    yv = np.reshape(yvals, (yvals.size, 1))

    O = np.ones((xv.size, yv.size))
    xy = xv*yv
    zvals = np.maximum(alpha*xy, beta*(O-xv)*(O-yv))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xvals, y=yvals, z = zvals))
    fig.update_layout(title = 'Reward Surface 3D Plot', width = 500, height = 500)

    return fig


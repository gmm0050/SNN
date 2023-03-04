
#Utilities for Data Type Conversion, Network slicing, etc...

import os
import os.path as osp
from pathlib import WindowsPath

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup as BS
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import imageio as m
#import cv2

import label
import coach

def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        dict = pickle.load(fo, encoding = 'bytes')

    return dict

def np2im(nparray):

    if len(nparray.shape) != 3:
        pilimage = Image.fromarray(nparray, mode = 'L')
    elif nparray.shape[0] == 1:
        nparray = np.squeeze(nparray)
        pilimage = Image.fromarray(nparray, mode = 'L')
    else:
        pilimage = Image.fromarray(nparray, mode = 'RGB')

    return pilimage


def rescale_label_np(nparray, lb, ub):

    return np.clip( np.around((nparray - lb)*(255 / (ub - lb))), 0, 255)



def slice_nn(net, node):



    return nodeout



def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

### BEGIN Load_Dataset ###


def load_dataset(datasetname, loadargs, imgtransforms, labelargs):

    #List of all currently available datasets
    alldatasetnames = ['imagenet', 'coco', 'mnist', 'cifar10', 'cifar100', 'pascal', 'openimages', 'objectnet', 'imagenet-p', 'imagenet-c', 'imagenet-a']

    #Initialize the list of sample dictionaries
    samplelist = []

    # Imagenet Dataset : Download images with annotations only
    if datasetname == 'imagenet':

        datapath = loadargs['datapath']
        newxsize = loadargs['resize'][0]
        newysize = loadargs['resize'][1]

        classdict = {}
        objlabels = []
        classlabels = []

        directories = os.listdir(datapath)

        for i in range(len(directories)):

            classdict[directories[i]] = i

        for dir in directories:

            classname = dir

            classpath = osp.join(datapath, dir)
            imgpath = osp.join(classpath, "Images/")
            annpath = osp.join(classpath, "Annotation/")

            imgfilenames = [f for f in os.listdir(imgpath) if osp.isfile(osp.join(imgpath, f))]

            #annfilenames = [f for f in os.listdir(annpath) if osp.isfile(osp.join(annpath, f))]
            print(dir)
            counter = 0
            for filename in imgfilenames:

                im = Image.open(osp.join(imgpath, filename))

                xsize, ysize = im.size
                xratio = newxsize/xsize
                yratio = newysize/ysize
                #
                counter += 1
                if counter == 1: 
                    print("Made it to the 2nd image!")
                    #newxsize = loadargs['resize'][0]

                t1 = T.ToTensor()
                imt = t1(im)

                if imt.shape[0] != 3:
                    continue
                
                imt = torch.squeeze(imt) 

                input = imgtransforms(imt)

                splitname = filename.split(".")

                imgname = splitname[0]

                annformat = labelargs['format']

                annname = imgname + "." + annformat

                #annfilename = [f for f in annfilenames if annname in annfilenames]

                annfilepath = osp.join(annpath, annname)

                numclasses = len(classdict)


                #Read in Annotations

                with open(annfilepath, 'r') as annfile:

                    data = annfile.read()

                Bdata = BS(data, annformat)

                objects = Bdata.find_all('object')

                size = Bdata.size 
                imgwidth = int(size.width.string)
                imgheight = int(size.height.string)

                labelshape = (imgheight, imgwidth, numclasses)

                #classname = Bdata.name

                classlabel = classdict[classname]
                
                centers = []

                objclasses = []

                for obj in objects:

                    clname = obj.find('name').string
                    cl = classdict[clname] ## Need to change classdict to have actual class names instead of directory names

                    istrunc = int(obj.truncated.string)
                    isdiff = int(obj.difficult.string)

                    bbox = obj.bndbox
                    xmin = int(bbox.xmin.string)
                    ymin = int(bbox.ymin.string)
                    xmax = int(bbox.xmax.string)
                    ymax = int(bbox.ymax.string)

                    bboxwidth = xmax - xmin
                    bboxheight = ymax - ymin
                    xloc = (xmax - xmin) / 2 + xmin
                    yloc = (ymax - ymin) / 2 + ymin

                    center = {}
                    center['xc'] = xratio*xloc
                    center['yc'] = yratio*yloc
                    center['cl'] = cl
                    center['isdiff'] = isdiff
                    center['istrunc'] = istrunc
                    center['width'] = xratio*bboxwidth
                    center['height'] = yratio*bboxheight

                    centers.append(center)

                    objclasses.append(cl)

                objlabels.append(set(objclasses))

                classlabels.append(set([classlabel]))

                #tgt_empty = torch.zeros(labelshape)

                #labeldict = label.make_label(tgt_empty, centers, classname, labelargs)

                #target = labeldict['label']

                samplelist.append({'input': input, 'label': centers})

        dataset = imagenet_annotated(samplelist)

    # COCO Dataset: find image_id for each image in its json file. find annotation json file with the same image_id
    elif datasetname == 'coco':
        
        datapath = loadargs['datapath']
        annpath = loadargs['annpath']
        annfilepath = osp.join(datapath, annpath)
        newxsize = loadargs['resize'][0]
        newysize = loadargs['resize'][1]

        classdict = {}
        objlabels = []
        classlabels = []

        #Load image json file. Find corresponding image file and annotation json file. Create Label using annotation file

        with open(annfilepath, 'r') as annfile:
            data = annfile.read()
        
        annotationformat = labelargs['format']

        Bdata = BS(data, annotationformat)

        categories = Bdata.find_all('categories')

        images = Bdata.find_all('images')

        annotations = Bdata.find_all("annotations")

        for category in categories:

            classdict[category['name']] = category['id']
        

        counter = 0

        for image in images:

            im = Image.open()
            
            xsize, ysize = im.size
            xratio = newxsize/xsize
            yratio = newysize/ysize

            counter += 1
            if counter == 1: 
                print("Made it to the 2nd image!")
                    #

            t1 = T.ToTensor()
            imt = t1(im)

            if imt.shape[0] != 3:
                continue
                
            imt = torch.squeeze(imt) #Remove possibility of unsqueezed image tensors messing up our transform procedure

            print(imt.shape) #Debugging shape mismatch when using transforms

            input = imgtransforms(imt)

            img_anns = annotations.find_all(image_id = image.image_id)

            centers = []

            objs = []

            for ann in img_anns:

                cl = ann.category_id

                bbox = ann.bbox

                xul = bbox[0]
                yul = bbox[1]
                width = bbox[2]
                height = bbox[3]

                xloc = xul + width/2
                yloc = yul + height/2

                isdiff = False
                istrunc = False

                center = {}
                center['xc'] = xratio*xloc
                center['yc'] = yratio*yloc
                center['cl'] = cl
                center['isdiff'] = isdiff
                center['istrunc'] = istrunc
                center['width'] = xratio*width
                center['height'] = yratio*height

                centers.append(center)

                objs.append(cl)

            objlabels.append(set(objs))

            classlabels.append(set(objs[0]))

            classname = None

            #labeldict = label.make_label(target, centers, classname, labelargs)

            #target = labeldict['label']

            samplelist.append({'input': input, 'label': centers})

        dataset = mscoco(samplelist)

    # MNIST Dataset: images and labels are stored in separate csv files that contain dictionaries with data / ids / labels
    elif datasetname == 'mnist':
        
        #Load from csv file into dictionary. Read data from dictionary. Read class label from dictionary. Put kernel at center.

        from mlxtend.data import loadlocal_mnist

        datapath = loadargs['datapath']
        annpath = loadargs['annpath']
        imgpath = loadargs['imgpath']
        annfilepath = osp.join(datapath, annpath)
        imgfilepath = osp.join(datapath, imgpath)

        classdict = {}
        objlabels = []
        classlabels = []

        X, y = loadlocal_mnist(images_path = imgfilepath, labels_path = annfilepath)

        for num in np.unique(y):

            classdict[str(num)] = num

        #centers = []

        for i in range(len(y)):
            
            imvec = X[i, :]

            imarray = np.reshape(imvec, (28, 28))

            img = np2im(imarray)

            if imgtransforms is not None:
                input = imgtransforms(img)
            else:
                t1 = T.ToTensor()
                input = t1(img)

            width = input.shape[-2] - 1
            height = input.shape[-1] - 1
            xloc = width/2 #Center of MNIST Image
            yloc = height/2
            cl = y[i]
            isdiff = False
            istrunc = False

            center = {}
            center['xc'] = xloc
            center['yc'] = yloc
            center['cl'] = cl
            center['isdiff'] = isdiff
            center['istrunc'] = istrunc
            center['width'] = width
            center['height'] = height

            #centers.append(center)

            objlabels.append(set([cl]))

            classlabels.append(set([cl]))

            #labeldict = label.make_label(centers, classdict, labelargs)

            #target = labeldict['label']

            samplelist.append({'input': input, 'label': [center]})

        dataset = mnist(samplelist)


    # CIFAR-10 Dataset: unpickle each batch file. extract image data from dict. extract image label from dict. Add kernel at image center.
    elif datasetname == 'cifar10':

        datapath = loadargs['datapath']
        annpath = osp.join(datapath, loadargs['annpath'])
        imgpath = osp.join(datapath, loadargs['imgpath'])

        classdict = {}
        objlabels = []
        classlabels = []

        #Unpickle batch files in loop. Read data from dict. Read label from dict, and add kernel at center.
        batchfiles = os.listdir(imgpath)

        metafilepath = osp.join(annpath, 'batches.meta')

        metadict = unpickle(metafilepath) 
        
        labelnames = metadict[b'label_names'] #List of all classnames in order

        #Make classdict
        for i in range(len(labelnames)):

            classdict[labelnames[i]] = i

        #Generate Inputs, Label Targets, and put them into a list for all images in the dataset we are loading
        centers = []

        for batchfile in batchfiles:

            batchdict = unpickle(osp.join(imgpath, batchfile))

            batchdata = batchdict[b'data']

            batchlabels = batchdict[b'labels']

            for i in range(len(batchlabels)):
                
                imvec = batchdata[i, :]
                imarray = np.reshape(imvec, (32, 32, 3))
                img = np2im(imarray)
                input = imgtransforms(img)

                xloc = 15.5
                yloc = 15.5
                cl = batchlabels[i]
                isdiff = False
                istrunc = False
                width = 31
                height = 31

                center = {}
                center['xc'] = xloc
                center['yc'] = yloc
                center['cl'] = cl
                center['isdiff'] = isdiff
                center['istrunc'] = istrunc
                center['width'] = width
                center['height'] = height

                #centers.append(center)

                objlabels.append(set([cl]))

                classlabels.append(set([cl]))

                #labeldict = label.make_label(centers, classdict, labelargs)

                #target = labeldict['label']

                samplelist.append({'input': input, 'label': center})

        dataset = cifar10(samplelist)

    # CIFAR-100 Dataset: unpickle each batch file. extract image data from dict. extract image label from dict. Add kernel at image center.
    elif datasetname == 'cifar100':
        
        #Same as CIFAR-10.

        datapath = loadargs['datapath']
        annpath = loadargs['annpath']
        imgpath = loadargs['imgpath']

        classdict = {}
        objlabels = []
        classlabels = []

        #Unpickle batch files in loop. Read data from dict. Read label from dict, and add kernel at center.
        batchfiles = os.listdir(imgpath)

        metafilepath = osp.join(annpath, 'batches.meta')

        metadict = unpickle(metafilepath) 
        
        labelnames = metadict[b'label_names'] #List of all classnames in order

        #Make classdict
        for i in range(len(labelnames)):

            classdict[labelnames[i]] = i

        #Generate Inputs, Label Targets, and put them into a list for all images in the dataset we are loading
        centers = []

        for batchfile in batchfiles:

            batchdict = unpickle(osp.join(imgpath, batchfile))

            batchdata = batchdict[b'data']

            batchlabels = batchdict[b'labels']

            for i in range(len(batchlabels)):
                
                imvec = batchdata[i, :]
                imarray = np.reshape(batchdata, (3, 32, 32))
                img = np2im(imarray)
                input = imgtransforms(img)

                xloc = 15.5
                yloc = 15.5
                cl = classdict[batchlabels[i]]
                isdiff = False
                istrunc = False
                width = 31
                height = 31

                center = {}
                center['xc'] = xloc
                center['yc'] = yloc
                center['cl'] = cl
                center['isdiff'] = isdiff
                center['istrunc'] = istrunc
                center['width'] = width
                center['height'] = height

                centers.append(center)

                objlabels.append(set([cl]))

                classlabels.append(set([cl]))

                #labeldict = label.make_label(centers, classdict, labelargs)

                #target = labeldict['label']

                samplelist.append({'input': input, 'label': centers})

        dataset = cifar100(samplelist)

    elif datasetname == 'pascal':
        
        #Import gluoncv. Read data, labels into list. Read data, and labels from list. Read class labels and bboxes from labels. Label target with bboxes/class_ids.
        import gluoncv.data as gdata
        import gluoncv.utils as gutils

        splits = loadargs['splits']

        classdict = {}
        objlabels = []
        classlabels = []


        pascaldataset = gdata.VOCDetection(splits = splits)

        for img, labelset in pascaldataset:

            img = topil(img)
            input = imgtransforms(img)

            bboxes = labelset[:, :4]
            classids = labelset[:, 4:5]

            centers = []
            objs = set(classids)

            for i in range(len(bboxes)):
                
                xmin = bboxes[i][0]
                ymin = bboxes[i][1]
                xmax = bboxes[i][2]
                ymax = bboxes[i][3]

                xloc = (xmax - xmin)/2 + xmin
                yloc = (ymax - ymin)/2 + ymin
                cl = classids[i]
                isdiff = False
                istrunc = False
                width = xmax - xmin
                height = ymax - ymin

                center = {}
                center['xc'] = xloc
                center['yc'] = yloc
                center['cl'] = cl
                center['isdiff'] = isdiff
                center['istrunc'] = istrunc
                center['width'] = width
                center['height'] = height

                centers.append(center)
            
            #labeldict = label.make_label(centers, classdict, labelargs)

            #target = labeldict['label']

            samplelist.append({'input': input, 'label': centers})

            objlabels.append(objs)

            classlabels.append(set(classids[0]))

        dataset = pascal(samplelist)

    elif datasetname == 'openimages':

        #Load Data with fiftyone package.
        raise NotImplementedError
        dataset = []

    elif datasetname == 'objectnet':

        #
        raise NotImplementedError
        dataset = []

    # Same as Imagenet but with extra specification of perturbation
    elif datasetname == 'imagenet-p':

        samplelist = []

    # Same as Imagenet but with extra specification of corruption
    elif datasetname == 'imagenet-c':

        samplelist = []

    # Same as Imagenet but with different images in each class 
    elif datasetname == 'imagenet-a':

        samplelist = []

    
    else:

        print("Error: No such dataset is known.")
        print("The available datasets are:")
        print("  ")
        print(alldatasetnames)


    return dataset, objlabels, classlabels, classdict


### END Load_Dataset ###


def find_annotation_imagenet(imfilename, annfilenames):

    exist_ann = False

    imfilecomps = imfilename.split("_")

    for filename in annfilenames:

        annfilecomps = filename.split("_")

        if imfilecomps[3] == annfilecomps[2]:

            exist_ann = True
            imname = imfilecomps[3]

    return exist_ann, imname




def split_batch_to_list(batchtensor, dim = 0):

    list = []

    split = torch.split(batchtensor, 1, dim=dim)

    for elem in split:

        list.append(elem)

    return list


def classify_with_threshold(testouts, testlabels, threshold):

    #testouts is a list of lists of class probabilities
    #testlabels is a list of sets of ground truth class indices (taken from annotations)
    #threshold is the probability threshold (0-1) above which we consider an object of a particular class to be present in the test image

    classset = set(range(len(testouts[0]))) #classset is a set that contains all of the indices corresponding to the possible classes

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for idx in range(len(testouts)):

        out = testouts[idx]

        testset = set([i for i in range(len(out)) if out[i] >= threshold]) #Take the indices of the classes with probabilities higher than the specified threshold

        truthset = testlabels[idx] #Transform labels for the ith test output into a set

        tpset = testset & truthset
        fpset = testset - tpset
        fnset = truthset - tpset
        tnset = classset - (testset | truthset)

        tp += len(tpset)
        fp += len(fpset)
        fn += len(fnset)
        tn += len(tnset)

    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn) #The same as true positive rate
    f1 = (2*prec*recall) / (prec + recall)

    return acc, tpr, fpr, prec, recall, f1, tp, fp, fn, tn



def classify_topn(testouts, testclasslabels, n):

    #testouts is a list of lists of class probabilities
    #testclasslabels is a list of sets of ground truth class indices
    #n is the number of classes to compare to truth

    tp = 0 #Total correct predictions
    total = len(testouts) #Total number of test samples

    for idx in range(total):

        out = testouts[idx]

        maxn = sorted(range(len(out)), key = lambda sub: out[sub])[-n:] #Take the n highest probabilities, sorted from lowest to highest

        maxnset = set(maxn)

        truthset = testclasslabels[idx]

        if len(maxnset & truthset) != 0:

            tp += 1
    
    acc = tp / total

    return acc

        
    


class imagenet_annotated(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class mscoco(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class cifar10(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class cifar100(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class mnist(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class pascal(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class openimagedataset(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class objectnet(Dataset):

    def __init__(self, samplelist):

        self.samples = samplelist

    def __len__(self):

        return len(self.samples)
        

    def __getitem__(self, idx):

        return self.samples[idx]


class cityscapesLoader(Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        ## Debug ##
        #print("image shape before")
        #print(img.shape)
        ##------------
        img = np.resize(img, (self.img_size[0], self.img_size[1], 3)) # uint8 with RGB mode
        ## Debug ##
        #print("image shape after")
        #print(img.shape)
        ##------------
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        ## Debug ##
        #print("label shape before")
        #print(lbl.shape)
        ##------------
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(np.resize(lbl, (self.img_size[0], self.img_size[1]))) 
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


class AddGaussianNoise(object):
    def __init__(self, mean = 0, std = 1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + 'mean = {}, std = {}'.format(self.mean, self.std)

class Clamp(object):
    def __init__(self, min = 0, max = 1):
        self.min = min
        self.max = max
    def __call__(self, tensor):
        return torch.clamp(tensor, min = self.min, max = self.max)
    def __repr__(self):
        return self.__class__.__name__ + 'min = {}, max = {}'.format(self.min, self.max)

class Enlarge(object):
    def __init__(self, size = [28, 28]):
        self.size = size
        #self.transform = T.functional.resize(size = self.size)
    def __call__(self, tensor):
        return T.functional.resize(tensor, size = self.size)
    def __repr__(self):
        return self.__class__.__name__ + 'size = ({}, {})'.format(self.size[0], self.size[1])


def is_inbbox(center, h, w, pos):

    inbbox = True

    if (pos[0] >= center[0] + h/2 or pos[0] <= center[0] - h/2) and (pos[1] >= center[1] + w or pos[1] <= center[1] - w):

        inbbox = False

    return inbbox

def run_tracking_test(coach, bgheight, bgwidth, numtracking):

    testouts = []
    testobjectives = []
    testprobs = []
    testtgts = []
    backgroundsize = (1, 1, bgheight, bgwidth)
    background = torch.zeros(backgroundsize)
    h = 28
    w = 28
    img = background
    target = torch.zeros(coach.testtargetshape)
    for n in range(numtracking):
        testdata = next(iter(coach.TestData))
        x1 = testdata['input']
        l1 = testdata['label']
        if type(l1) is list:
            batchlabel1 = l1[0]
        elif type(l1) is dict:
            batchlabel1 = l1
        testdata = next(iter(coach.TestData))
        x2 = testdata['input']
        l2 = testdata['label']
        if type(l2) is list:
            batchlabel2 = l2[0]
        elif type(l2) is dict:
            batchlabel2 = l2

        rand1xy = torch.randint(14, 126, (2,))
        rand2xy = rand1xy
        while is_inbbox(rand1xy, h, w, rand2xy):
            rand2xy = torch.randint(14, 126, (2,))
        img[:, :, rand1xy[0]-14:rand1xy[0]+14, rand1xy[1] - 14: rand1xy[1]+14] = x1
        img[:, :, rand2xy[0]-14:rand2xy[0]+14, rand2xy[1] - 14: rand2xy[1]+14] = x2
        target = label.batch_label(target, batchlabel1, coach.labelargs)
        target = label.batch_label(target, batchlabel2, coach.labelargs)
        if coach.labelargs['use_nullclass'] == False:
            target = target + 0.015
            target = torch.clamp(target, min = 0.0, max = 1.0)
            #target = target + 0.015

        ds_target = T.Resize(bgheight/coach.base_scales[0], bgwidth/coach.base_scales[0])
        #ds_input = T.Resize(bgheight, bgwidth)
        target = ds_target(target)
        #x = ds_input(img)
        x = img

    
    testobjs = []

    with torch.no_grad():
                
        for s, scale in enumerate(coach.base_scales):
                    
            #self.net.set_scale_factors(scale)
                    
            out, target = coach.forwardpass(x, target)

            testobjs.append(coach.calc_objective(out, target))

            outs = split_batch_to_list(out, dim=0)

            probabilities = coach.classify(outs)


            testobjectives[-1].append(max(testobjs))

            testprobs[-1].append(probabilities)

            testouts[-1].append(outs)
            testtgts[-1].append(target)

    return testouts, testobjectives, testprobs, testtgts

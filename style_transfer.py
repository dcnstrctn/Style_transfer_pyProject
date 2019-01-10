# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import PIL

import numpy as np


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=512):
    transform = T.Compose([
        T.Scale(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

dtype = torch.FloatTensor

# Load the pre-trained SqueezeNet model.
cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)

#don't want to train the model any further, so we don't want PyTorch to waste computation 
# computing gradients on parameters we're never going to update.
for param in cnn.parameters():
    param.requires_grad = False

def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    """
    _, C, H, W = content_current.size()

    F = content_current.view(C, H*W)
    P = content_original.view(C, H*W)
    
    loss = content_weight * (torch.sum((F - P)**2))
    
    return loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    """
    N, C, H, W = features.size()
    F = features.view(N, C, H*W) 
    gram = torch.mm(F[0,:,:], F[0,:,:].transpose(1,0))
    if normalize == True:
        gram /=(H*W*C)
    gram = gram.unsqueeze(0)# Add back first dimension
    
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    """
    style_loss = Variable(torch.zeros(1))
    for i, layer in enumerate(style_layers):
        style_loss += style_weights[i] * torch.sum((gram_matrix(feats[layer]) - style_targets[i])**2)
        
    return style_loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    """
    loss = tv_weight * (torch.sum((img[:,:,:,1:]-img[:,:,:,:-1])**2)+torch.sum((img[:,:,1:,:]-img[:,:,:-1,:])**2))
    return loss

def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1)
    else:
        img = content_img.clone().type(dtype)

    img_var = Variable(img, requires_grad=True)

    #optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)
    
    for t in range(200):
        if t < 190:
            img.clamp_(-1.5, 1.5)
        optimizer.zero_grad()
        feats = extract_features(img_var, cnn)
        
        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img_var, tv_weight) 
        loss = c_loss + s_loss + t_loss
        
        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img_var], lr=decayed_lr)
        optimizer.step()

    return deprocess(img.cpu())

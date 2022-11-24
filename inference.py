import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

def seperate_images(image, matte, save_dir, file_name):
    # calculate display resolution
    w, h = image.width, image.height

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
        
    matte = np.repeat(matte[:, :, None], 3, axis=2)
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
    background = image * (1 - matte) + np.full(image.shape, 255) * matte
    Image.fromarray(np.uint8(foreground)).save(os.path.join(save_dir, 'foreground', file_name))
    Image.fromarray(np.uint8(background)).save(os.path.join(save_dir, 'background', file_name))
    
    # combine image, foreground, and alpha into one line    
    rw, rh = 800, int(h * 800 / (3 * w))    
    combined = np.concatenate((image, foreground, matte * 255), axis=1)
    Image.fromarray(np.uint8(combined)).resize((rw, rh)).save(os.path.join(save_dir, 'combined', file_name))

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='input_images', help='path of input images')
    parser.add_argument('--output-path', type=str, default='output_images', help='path of output images')
    parser.add_argument('--ckpt-path', type=str, default='pretrained/matting.ckpt', help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()
        
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path,'matte'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path,'foreground'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path,'background'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path,'combined'), exist_ok=True)

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        input_image = Image.open(os.path.join(args.input_path, im_name))

        # unify image channels to 3
        im = np.asarray(input_image)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        out_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path,'matte',out_name))
        seperate_images(input_image,matte,args.output_path,out_name)

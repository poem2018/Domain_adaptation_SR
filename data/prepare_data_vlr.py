import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time
import h5py

import torch
import torch.nn.functional as F

import h5py
import fastmri
from fastmri.data import transforms as T
from matplotlib import pyplot as plt

import imageio
import numpy as np

# 256 - 512 - 1024
# 64 - 256 - 1024
# 16 - 128 - 1024

# def esr_downsample():
#python data/prepare_data.py  --path ../../dataset/CelebAMask-HQ/CelebA-HQ-img/  --out ./dataset/Celeba --size 256,1024 -l
#python data/prepare_data_vlr.py  --path ../../dataset/CelebAMask-HQ/CelebA-HQ-img/  --out ./dataset/Celeba_2x --size 512,1024,256 -l


##use esrgan to get vlr_64, rename to lr_64, lr_256 ->hr_256
#python scripts/generate_multiscale_DF2K.py --input ../SR3/dataset/Celeba_256_1024/lr_256 --output ../SR3/dataset/Celeba_256_1024/vlr_64

##split the train test data
##to get vlr -> sr_vlr  resize_worker





def save_test_resize_lr(path, sizes, lmdb_save):
    print("start")
    image_files = [f for f in os.listdir(path)]
    # loaded_images = []
    print("path array finish")

    # Load each image
    for image_file in tqdm(image_files, desc="Loading images"):
        image_path = os.path.join(path, image_file)  
        img = Image.open(image_path)
        img_idx = image_path.split("/")[-1]
        img = img.convert('RGB')
        if sizes[0]==img.size[0]:
            out = resize_and_convert(img, size=sizes[1], resample=Image.BICUBIC)
            out.save('../dataset/finetune_8x/lr_{}/{}'.format(sizes[1], img_idx))
            out_r = resize_and_convert(out, size=sizes[0], resample=Image.BICUBIC)
            out_r.save('../dataset/finetune_8x/sr_{}_{}/{}'.format(sizes[1], sizes[0], img_idx))
            # out.save('./dataset/test/sample_64.png')
        else:
            print("wrong size")

    return True

def crop(image):
    if np.shape(image)[0]!=np.shape(image)[1]:
        ###crop it to be same
        crop_size = min(np.shape(image)[0], np.shape(image)[1])
        #center crop
        image = image.crop((0, 0, crop_size, crop_size))
    return image

def save_coco(sizes, ratio = [0.8, 0.1, 0.1]):
    image_path = "../dataset/COCO/unlabeled2017"
    #load the image from this folder
    img_data = [f for f in os.listdir(image_path)]

    print("start")
    np.random.shuffle(img_data)
    data_len = len(img_data)
    train_data = img_data[:int(data_len*ratio[0])]
    test_data = img_data[int(data_len*ratio[0]):int(data_len*(ratio[0]+ratio[1]))]
    valid_data = img_data[int(data_len*(ratio[0]+ratio[1])):]

    print(len(train_data), len(test_data), len(valid_data))
    i=0
    for data in tqdm(train_data):
        ##### data is a file name, read and load it
        data = Image.open(os.path.join(image_path, data))
        img = crop(data)
        img = np.array(img)
        # img = img.convert('RGB')
        
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=False, flag='coco')
        # import pdb;pdb.set_trace()
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        path = '../dataset/COCO/train'
        if not os.path.exists(path):
            os.makedirs(path)
            os.mkdir(path + '/lr_{}'.format(sizes[2]))
            os.mkdir(path + '/hr_{}'.format(sizes[0]))
            os.mkdir(path + '/sr_{}_{}'.format(sizes[2], sizes[0]))

        #save np array as image
        plt.imsave(path + '/lr_{}/{}.png'.format(sizes[2], i), vlr_img, cmap='gray')
        plt.imsave(path + '/hr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[2], sizes[0], i), sr_img, cmap='gray')
        i+=1
    
    for data in tqdm(valid_data):
        data = Image.open(os.path.join(image_path, data))
        img = crop(data)
        img = np.array(img)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=False, flag='coco')
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        path = '../dataset/COCO/valid'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/lr_{}'.format(sizes[2]))
            os.makedirs(path + '/hr_{}'.format(sizes[0]))
            os.makedirs(path + '/sr_{}_{}'.format(sizes[2], sizes[0]))

        plt.imsave(path + '/lr_{}/{}.png'.format(sizes[2], i), vlr_img, cmap='gray')
        plt.imsave(path + '/hr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[2], sizes[0], i), sr_img, cmap='gray')
        i+=1

    
    for data in tqdm(test_data):
        data = Image.open(os.path.join(image_path, data))
        img = crop(data)
        img = np.array(img)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test = True, flag='coco')
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        if sizes[0]==64:
            path = '../dataset/COCO/test_4x'
        if sizes[0]==32:
            path = '../dataset/COCO/test_8x'
        if sizes[0]==128:
            path = '../dataset/COCO/test_2x'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/lr_{}'.format(sizes[0]))
            os.makedirs(path + '/hr_{}'.format(sizes[1]))
            os.makedirs(path + '/sr_{}_{}'.format(sizes[0], sizes[1]))

        plt.imsave(path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        i+=1

    print("sample")
    return True



def save_fastmri(sizes, ratio = [0,0,0]):
    file_paths = '../../../datasets/fastMRI/multicoil_val/T2'  # Add your file paths here
    file_name_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.h5')]

    print("start")

    idx = 0
    for i in tqdm(file_name_list):
        slice_idx = 7
        file_name = i
        
        # try:
        #     with h5py.File(file_name, 'r') as file:
        #         # Perform read operations
        #         print(list(file.keys()))
        # except OSError as e:
        print(file_name)
        ##if unable to read h5 file print file name

        hf = h5py.File(file_name)
        # print(file_name)

        volume_kspace = hf['kspace'][()]
        slice_kspace = volume_kspace[slice_idx]
        slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
        slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
        slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

        slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
        image = np.abs(slice_image_rss.numpy())
        ##add center crop to make resolution become 320x320
        image = image[220:540, 40:360]

        # image = (image - image.min()) / (image.max() - image.min()) 
        if sizes[0]==64:
            path = '../dataset/fastMRI/T2_npy_val_4x/'
        if sizes[0]==32:
            path = '../dataset/fastMRI/T2_npy_val_8x/'
        if sizes[0]==128:
            path = '../dataset/fastMRI/T2_npy_val_2x/'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/lr_{}'.format(sizes[0]))
            os.makedirs(path + '/hr_{}'.format(sizes[1]))
            os.makedirs(path + '/sr_{}_{}'.format(sizes[0], sizes[1]))
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(image, sizes=sizes, lmdb_save=False, test=True, flag='fast')
        
        # plt.imsave('./sample.png', lr_img, cmap='gray')
        # plt.imsave(path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        # plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        # print(lr_img.max(), lr_img.min(), hr_img.max(), hr_img.min(), sr_img.max(), sr_img.min())

        # lr_img = lr_img.astype(np.float32)
        # hr_img = hr_img.astype(np.float32)
        # sr_img = sr_img.astype(np.float32)
        ##chanegt the image from (128x128) to (128x128x3), copy the channel dim for three time
        lr_img = np.stack((lr_img,)*3, axis=-1)
        hr_img = np.stack((hr_img,)*3, axis=-1)
        sr_img = np.stack((sr_img,)*3, axis=-1)
        #save as npy array
        np.save(path + '/lr_{}/{}.npy'.format(sizes[0], idx), np.array(lr_img))
        np.save(path + '/hr_{}/{}.npy'.format(sizes[1], idx), np.array(hr_img))
        np.save(path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], idx), np.array(sr_img))
        print(lr_img.max(), lr_img.min(), hr_img.max(), hr_img.min(), sr_img.max(), sr_img.min())
        idx+=1

        # import pdb;pdb.set_trace()
        # test = np.load(path + '/lr_{}/{}.npy'.format(sizes[0], i))


    return True


            
def save_tumor_crop():
    file_paths = '../dataset/notumor/notumor_copy'  # Add your file paths here
    output_dir = '../dataset/notumor_crop/'
    print("start")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    idx = 0
    for filename in os.listdir(file_paths):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(file_paths, filename)
            with Image.open(img_path) as img:
                if img.size[0] != img.size[1]:
                    min = np.min(img.size)
                    left = (img.size[0] - min) / 2
                    top = (img.size[1] - min) / 2
                    right = (img.size[0] + min) / 2
                    bottom = (img.size[1] + min) / 2
                    img = img.crop((left, top, right, bottom))
                    
                if img.size[0] >= 256:
                    # Resize the image to 256x256
                    img_resized = img.resize((256, 256), Image.BILINEAR)
                    # Save the image as PNG
                    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
                    img_resized.save(output_path, 'PNG')
    return True

def tumor_crop_resize(sizes):
    file_paths = '../dataset/notumor_crop'  # Add your file paths here
    file_name_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.png')]

    print("start")
    idx = np.arange(0, len(file_name_list))
    np.random.shuffle(idx)
    val_idx = idx[:170]
    train_idx = idx[170:]

    # image = (image - image.min()) / (image.max() - image.min()) 
    if sizes[0]==64:
        val_path = '../dataset/notumor_crop_split/val_4x/'
        train_path = '../dataset/notumor_crop_split/train_4x/'
    if sizes[0]==32:
        val_path = '../dataset/notumor_crop_split/val_8x/'
        train_path = '../dataset/notumor_crop_split/train_8x/'
    if sizes[0]==128:
        val_path = '../dataset/notumor_crop_split/val_2x/'
        train_path = '../dataset/notumor_crop_split/train_2x/'
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        os.makedirs(val_path + '/lr_{}'.format(sizes[0]))
        os.makedirs(val_path + '/hr_{}'.format(sizes[1]))
        os.makedirs(val_path + '/sr_{}_{}'.format(sizes[0], sizes[1]))
        os.makedirs(train_path)
        os.makedirs(train_path + '/lr_{}'.format(sizes[0]))
        os.makedirs(train_path + '/hr_{}'.format(sizes[1]))
        os.makedirs(train_path + '/sr_{}_{}'.format(sizes[0], sizes[1]))

    i=0
    for idx in tqdm(val_idx):
        data = Image.open(file_name_list[idx])
        img = np.array(data)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=True, flag='notumor')
        # np.save(val_path + '/lr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        # np.save(val_path + '/hr_{}/{}.npy'.format(sizes[1], i), np.array(hr_img))
        # np.save(val_path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], i), np.array(sr_img))
        plt.imsave(val_path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(val_path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        plt.imsave(val_path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        i+=1
    
    for idx in tqdm(train_idx):
        data = Image.open(file_name_list[idx])
        img = np.array(data)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=True, flag='notumor')
        # np.save(train_path + '/lr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        # np.save(train_path + '/hr_{}/{}.npy'.format(sizes[1], i), np.array(hr_img))
        # np.save(train_path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], i), np.array(sr_img))
        plt.imsave(train_path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(train_path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        plt.imsave(train_path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        i+=1
    

    # import pdb;pdb.set_trace()
    # test = np.load(path + '/lr_{}/{}.npy'.format(sizes[0], i))
    return True


#######################################################3
def oasis_save():
    folder_path = '../dataset/Data/Non Demented/'  # Add your file paths here

    count=0
    # Loop through the files in the folder
    for filename in os.listdir(folder_path):

        
        if "157" in filename:  # Check if the file name contains '157'
            file_path = os.path.join(folder_path, filename)
            print(count)
            count+=1
            
            # Open the image file
            with Image.open(file_path) as img:
                # Resize the image to 256x256
                img_resized = img.resize((256, 256))
                
                # Save the resized image, overwriting the original
                img_resized.save("./../dataset/Data/proceed/nd{}.png".format(count), 'PNG')

    print("Resizing completed.")


##157
def oasis_crop_resize(sizes):
    file_paths = '../dataset/Data/proceed/'  # Add your file paths here
    file_name_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.png')]

    print("start")
    idx = np.arange(0, len(file_name_list))
    np.random.shuffle(idx)
    val_idx = idx[:60]
    train_idx = idx[60:360]

    # image = (image - image.min()) / (image.max() - image.min()) 
    if sizes[0]==64:
        val_path = '../dataset/oasis_crop_split/val_4x/'
        train_path = '../dataset/oasis_crop_split/train_4x/'
    if sizes[0]==32:
        val_path = '../dataset/oasis_crop_split/val_8x/'
        train_path = '../dataset/oasis_crop_split/train_8x/'
    if sizes[0]==128:
        val_path = '../dataset/oasis_crop_split/val_2x/'
        train_path = '../dataset/oasis_crop_split/train_2x/'
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        os.makedirs(val_path + '/lr_{}'.format(sizes[0]))
        os.makedirs(val_path + '/hr_{}'.format(sizes[1]))
        os.makedirs(val_path + '/sr_{}_{}'.format(sizes[0], sizes[1]))
        os.makedirs(train_path)
        os.makedirs(train_path + '/lr_{}'.format(sizes[0]))
        os.makedirs(train_path + '/hr_{}'.format(sizes[1]))
        os.makedirs(train_path + '/sr_{}_{}'.format(sizes[0], sizes[1]))

    i=0
    for idx in tqdm(val_idx):
        data = Image.open(file_name_list[idx])
        img = np.array(data)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=True, flag='notumor')
        # np.save(val_path + '/lr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        # np.save(val_path + '/hr_{}/{}.npy'.format(sizes[1], i), np.array(hr_img))
        # np.save(val_path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], i), np.array(sr_img))
        plt.imsave(val_path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(val_path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        plt.imsave(val_path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        i+=1
    
    for idx in tqdm(train_idx):
        data = Image.open(file_name_list[idx])
        img = np.array(data)
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(img, sizes=sizes, lmdb_save=False, test=True, flag='notumor')
        # np.save(train_path + '/lr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        # np.save(train_path + '/hr_{}/{}.npy'.format(sizes[1], i), np.array(hr_img))
        # np.save(train_path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], i), np.array(sr_img))
        plt.imsave(train_path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        plt.imsave(train_path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        plt.imsave(train_path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')
        i+=1
    

    # import pdb;pdb.set_trace()
    # test = np.load(path + '/lr_{}/{}.npy'.format(sizes[0], i))
    return True




##split the numpy array data into train test valid ratio as 0.9/0.1/0.1, random shuffle the data
def save_IXI(img_data, sizes, ratio = [0.8, 0.1, 0.1]):
    print("start")
    np.random.shuffle(img_data)
    train_data = img_data[:int(img_data.shape[0]*ratio[0])]
    test_data = img_data[int(img_data.shape[0]*ratio[0]):int(img_data.shape[0]*(ratio[0]+ratio[1]))]
    valid_data = img_data[int(img_data.shape[0]*(ratio[0]+ratio[1])):]

    # train_data = train_data[:2]
    # test_data = test_data[:2]
    # valid_data = valid_data[:2]
    #rgb range to [-1, 1]
    import matplotlib.pyplot as plt
    # train_data = train_data [0]
    # train_data_n = (train_data - train_data.min()) / (train_data.max() - train_data.min())

    # import pdb;pdb.set_trace()

    i=0
    for data in tqdm(train_data):
        print("1",data.max(), data.min())
        # data = (data - data.min()) / (data.max() - data.min())
        # img = Image.fromarray(data)
        # img = img.convert('RGB')
        
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(data, sizes=sizes, lmdb_save=False, test=False, flag='ixi')
        # import pdb;pdb.set_trace()
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        if sizes[0]==64:
            path = '../dataset/IXI_T2_train_npy_4x'
        if sizes[0]==32:
            path = '../dataset/IXI_T2_train_npy_8x'
        if sizes[0]==128:
            path = '../dataset/IXI_T2_train_npy_2x'
        if not os.path.exists(path):
            os.makedirs(path)
            os.mkdir(path + '/lr_{}'.format(sizes[2]))
            os.mkdir(path + '/hr_{}'.format(sizes[0]))
            os.mkdir(path + '/sr_{}_{}'.format(sizes[2], sizes[0]))

        #save np array as image
        np.save(path + '/lr_{}/{}.npy'.format(sizes[2], i), np.array(vlr_img))
        np.save(path + '/hr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        np.save(path + '/sr_{}_{}/{}.npy'.format(sizes[2], sizes[0], i), np.array(sr_img))

        # plt.imsave(path + '/lr_{}/{}.png'.format(sizes[2], i), vlr_img, cmap='gray')
        # plt.imsave(path + '/hr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        # plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[2], sizes[0], i), sr_img, cmap='gray')
        i+=1

    
    for data in tqdm(valid_data):
        # data = (data - data.min()) / (data.max() - data.min())
        # img = Image.fromarray(data)
        # img = img.convert('RGB')
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(data, sizes=sizes, lmdb_save=False, test=False, flag='ixi')
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        if sizes[0]==64:
            path = '../dataset/IXI_T2_val_npy_4x'
        if sizes[0]==32:
            path = '../dataset/IXI_T2_val_npy_8x'
        if sizes[0]==128:
            path = '../dataset/IXI_T2_val_npy_2x'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/lr_{}'.format(sizes[2]))
            os.makedirs(path + '/hr_{}'.format(sizes[0]))
            os.makedirs(path + '/sr_{}_{}'.format(sizes[2], sizes[0]))

        np.save(path + '/lr_{}/{}.npy'.format(sizes[2], i), np.array(vlr_img))
        np.save(path + '/hr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        np.save(path + '/sr_{}_{}/{}.npy'.format(sizes[2], sizes[0], i), np.array(sr_img))

        # plt.imsave(path + '/lr_{}/{}.png'.format(sizes[2], i), vlr_img, cmap='gray')
        # plt.imsave(path + '/hr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        # plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[2], sizes[0], i), sr_img, cmap='gray')
        i+=1
    
    for data in tqdm(test_data):
        # data = (data - data.min()) / (data.max() - data.min())
        # img = Image.fromarray(data)
        # img = img.convert('RGB')
        lr_img, hr_img, sr_img, vlr_img = resize_vlr(data, sizes=sizes, lmdb_save=False, test = True, flag='ixi')
        ##add train test valid split here with ratio as 0.9/0.1/0.1
        # path = '../dataset/IXI_T2_test_npy'
        if sizes[0]==64:
            path = '../dataset/IXI_T2_test_npy_4x'
        if sizes[0]==32:
            path = '../dataset/IXI_T2_test_npy_8x'
        if sizes[0]==128:
            path = '../dataset/IXI_T2_test_npy_2x'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/lr_{}'.format(sizes[0]))
            os.makedirs(path + '/hr_{}'.format(sizes[1]))
            os.makedirs(path + '/sr_{}_{}'.format(sizes[0], sizes[1]))

        np.save(path + '/lr_{}/{}.npy'.format(sizes[0], i), np.array(lr_img))
        np.save(path + '/hr_{}/{}.npy'.format(sizes[1], i), np.array(hr_img))
        np.save(path + '/sr_{}_{}/{}.npy'.format(sizes[0], sizes[1], i), np.array(sr_img))

        # plt.imsave(path + '/lr_{}/{}.png'.format(sizes[0], i), lr_img, cmap='gray')
        # plt.imsave(path + '/hr_{}/{}.png'.format(sizes[1], i), hr_img, cmap='gray')
        # plt.imsave(path + '/sr_{}_{}/{}.png'.format(sizes[0], sizes[1], i), sr_img, cmap='gray')

        i+=1


    print("sample")
    return True



##write a function that read different resolution image and save it as npy array
def save_multi():
    root_path = '../dataset/COCO/'
    list = ['train/hr_random', 'train/lr_random', 'train/sr_random', \
            'valid/hr_random', 'valid/lr_random', 'valid/sr_random', \
            'test_random/hr', 'test_random/lr', 'test_random/sr']
    for items in list:
        path = os.path.join(root_path, items)
        ##check if file exist, if exist, clear it
        if os.path.exists(path):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
        else:
            os.mkdir(path)

    ##read the file under folder '../dataset/COCO/train/'
    root_path = ['../dataset/COCO/train/', '../dataset/COCO/valid/']
    for path in root_path:
        folder_hr = [ 'hr_32', 'hr_64', 'hr_128']
        folder_lr = [ 'lr_4', 'lr_16', 'lr_64']
        folder_sr = [ 'sr_4_32','sr_16_64', 'sr_64_128']
        print(folder_hr, folder_lr, folder_sr)

        file_hr = [f for f in os.listdir(path + folder_hr[0])]
        file_lr = [f for f in os.listdir(path + folder_lr[0])]
        file_sr = [f for f in os.listdir(path + folder_sr[0])]

        ##generate random number list that is 1/3 of the len of the number of file_list
        random_list = np.arange(len(file_hr))
        random_list = np.array_split(random_list, 3)

        for idx in range(3):
            ##tqdm
            for j in tqdm(random_list[idx]):
                ##read the image file under folder '../dataset/COCO/train/' + folder

                hr= Image.open(path + folder_hr[idx] + '/' + file_hr[j])
                lr= Image.open(path + folder_lr[idx] + '/' + file_lr[j])
                sr= Image.open(path + folder_sr[idx] + '/' + file_sr[j])
                ##save the image file under folder '../dataset/COCO/train/' + folder
                if file_hr[j] in os.listdir(path + '/hr_random'):
                    print("repeat!!!!!")
                    import pdb;pdb.set_trace()
                hr.save(path + '/hr_random/' +  file_hr[j])  #folder_hr[idx] + '_' +
                lr.save(path + '/lr_random/' +  file_lr[j])  #folder_lr[idx] + '_' +
                sr.save(path + '/sr_random/' +  file_sr[j])  #folder_sr[idx] + '_' +


    return True

    



def resize_vlr(img, sizes, lmdb_save, test, flag):
    # img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, flag=flag, resample=Image.BICUBIC, lmdb_save=lmdb_save, test =test)
    lr_img, hr_img, sr_img, vlr_img = out
    return lr_img, hr_img, sr_img, vlr_img




#########resize the grey image#####3
def resize_and_convert(img, size, resample, flag = None):
    
    if flag =="coco" :
        image = Image.fromarray(img.astype(np.uint8))  
    if flag =='fast':
        image = Image.fromarray(img)  
    if flag == "ixi":
        image = Image.fromarray(img)    #ixi
    if flag == 'notumor':
        image = Image.fromarray(img.astype(np.uint8)) 

    resize_img = np.array(image.resize((size, size), Image.BICUBIC))

    ##normalzie for grey image
    if flag =='fast' or flag == "ixi"or flag == "notumor":
        resize_img = (resize_img - resize_img.min()) / (resize_img.max() - resize_img.min())
    # print("4",resize_img.max(), resize_img.min())

    return resize_img



###resize natural image#####
# def resize_and_convert(img, size, resample):
   
#     if(img.size[0] != size):
#         # img = trans_fn.resize(img, size, resample)
#         # img = trans_fn.center_crop(img, size)
#         img = np.array(img.resize(size, Image.BICUBIC))
#         img = img.resize(size, resample)
#         img = img.center_crop(img, size)
#     return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128, 8), flag=None, resample=Image.BICUBIC, lmdb_save=False, test = False):
    lr_img = resize_and_convert(img, sizes[0], resample, flag)
    hr_img = resize_and_convert(img, sizes[1], resample, flag)
    vlr_img = resize_and_convert(lr_img, sizes[2], resample, flag)
    if test == False:
        sr_img = resize_and_convert(vlr_img, sizes[0], resample, flag)  #need modify
    else:
        if flag == 'notumor':
            lr_img = (lr_img * 255).astype(np.uint8)
        sr_img = resize_and_convert(lr_img, sizes[1], resample, flag)
    # if test == False:
    #     # use avg pooling to resize the image
    #     image_tensor = torch.tensor(vlr_img).unsqueeze(0).unsqueeze(0).float()
    #     sr_img = F.interpolate(image_tensor, size=(sizes[0], sizes[0]), mode='nearest')
    # else:
    #     image_tensor = torch.tensor(lr_img).unsqueeze(0).unsqueeze(0).float()
    #     sr_img = F.interpolate(image_tensor, size=(sizes[1], sizes[1]), mode='nearest')
    # sr_img = np.array(sr_img[0][0])

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)
        vlr_img = image_convert_bytes(vlr_img)

    return [lr_img, hr_img, sr_img, vlr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)
    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img, vlr_img = imgs
        if not wctx.lmdb_save:
            vlr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[2], i.zfill(5)))
            lr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[2], wctx.sizes[0], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[2], i.zfill(5)).encode('utf-8'), vlr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[2], wctx.sizes[0], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128, 8), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[2]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[2], sizes[0]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        idx_end = 0
        for file in tqdm(files):
            i = file.name
            path1 = '{}/hr_{}/{}'.format(out_path, sizes[0], i)
            path2 = '{}/lr_{}/{}'.format(out_path, sizes[2], i)
            path3 = '{}/sr_{}_{}/{}'.format(out_path, sizes[2], sizes[0], i)
            if not os.path.exists(path1) or not os.path.exists(path2) or not os.path.exists(path3):
                print(path1,path2,path3)
                i, imgs = resize_fn(file)
                lr_img, hr_img, sr_img, vlr_img = imgs
                if not lmdb_save:
                    lr_img.save(
                        '{}/hr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                    vlr_img.save(
                        '{}/lr_{}/{}.png'.format(out_path, sizes[2], i.zfill(5)))
                    sr_img.save(
                        '{}/sr_{}_{}/{}.png'.format(out_path, sizes[2], sizes[0], i.zfill(5)))
                else:
                    with env.begin(write=True) as txn:
                        txn.put('hr_{}_{}'.format(
                            sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                        txn.put('lr_{}_{}'.format(
                            sizes[2], i.zfill(5)).encode('utf-8'), vlr_img)
                        txn.put('sr_{}_{}_{}'.format(
                            sizes[2], sizes[0], i.zfill(5)).encode('utf-8'), sr_img)
                total += 1
                print(total)
                if lmdb_save:
                    with env.begin(write=True) as txn:
                        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))



if __name__ == '__main__':
    print("starttttt")

    
    # save_coco(sizes = ( 128, 256, 64))
    # save_coco(sizes = ( 64, 256, 16))
    # save_coco(sizes = ( 32, 256, 4))


    # img_data = np.load("../dataset/IXI_T2.npy")
    # print("load down")
    # save_IXI(img_data, sizes = ( 128, 256, 64))
    # save_IXI(img_data, sizes = ( 64, 256, 16))
    # save_IXI(img_data, sizes = ( 32, 256, 4))

    # save_fastmri(sizes = ( 128, 256, 64))
    # save_fastmri(sizes = ( 64, 256, 16))
    # save_fastmri(sizes = ( 32, 256, 4))


    # save_tumor_crop()
    # tumor_crop_resize( ( 64, 256, 16))

    # oasis_save()
    oasis_crop_resize( ( 64, 256, 16))

    # save_multi()



    # save_test_resize_lr('./dataset/test_Celeba_2x/lr_512/', sizes = (512, 1024), lmdb_save=True) 
    # save_test_resize_lr('../dataset/finetune_8x/lr_128/', sizes = (128,16), lmdb_save=True) 


    # t0 = np.load('/u/ruikez2/ruikez2/ruike/SR_Diffusion/dataset/fastMRI/T2_image_finetune/multicoil_T2_test/hr_256/0.npy')
    # t1 = np.load('/u/ruikez2/ruikez2/ruike/SR_Diffusion/dataset/fastMRI/T2_image_finetune/multicoil_T2_test/hr_256/1.npy')
    # ##show npy array as png
    # plt.imsave('./test0.png', t0, cmap='gray')
    # plt.imsave('./test1.png', t1, cmap='gray')
    # import pdb;pdb.set_trace()

    


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', '-p', type=str,
#                         default='{}/Dataset/celebahq_256'.format(Path.home()))
#     parser.add_argument('--out', '-o', type=str,
#                         default='./dataset/celebahq')

#     parser.add_argument('--size', type=str, default='64,512')
#     parser.add_argument('--n_worker', type=int, default=3)
#     parser.add_argument('--resample', type=str, default='bicubic')
#     # default save in png format
#     parser.add_argument('--lmdb', '-l', action='store_true')   ##store_true

#     args = parser.parse_args()

#     resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
#     resample = resample_map[args.resample]
#     sizes = [int(s.strip()) for s in args.size.split(',')]

#     # import pdb;pdb.set_trace()
#     args.out = '{}_{}_{}_{}'.format(args.out, sizes[0], sizes[1], sizes[2])
#     prepare(args.path, args.out, args.n_worker,
#             sizes=sizes, resample=resample, lmdb_save=args.lmdb)

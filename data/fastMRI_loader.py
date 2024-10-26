import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import os


import h5py
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
from PIL import Image
import torch.nn.functional as F

##write a 
file_paths = '../../../dataset/fastMRI/multicoil_train/T2'  # Add your file paths here
file_name_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.h5')]


for i in range(1): #len(file_name_list)):
    file_name = file_name_list[i]
    hf = h5py.File(file_name)

    volume_kspace = hf['kspace'][()]
    for j in range(volume_kspace.shape[0]):
        print(i,j)
        slice_kspace = volume_kspace[j]  

        slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
        slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
        slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image


        slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
        image = np.abs(slice_image_rss.numpy())
        ##add center crop to make resolution become 320x320
        image = image[220:540, 40:360]

        #normalize the image to 0-1
        image = (image - image.min()) / (image.max() - image.min()) 

        # plt.imshow(image, cmap='gray')
        # plt.savefig(f'../../../dataset/fastMRI/multicoil_train/T2_image/{i}_{j}.png')

        #save the image as npy
        np.save(f'../../../dataset/fastMRI/multicoil_train/T2_image/{i}_{j}.npy', image)


        # import pdb; pdb.set_trace()
        # image = Image.fromarray(image)  # 'L' mode is for (8-bit pixels, black and white)
        # image.save(f'../../../dataset/fastMRI/multicoil_train/T2_image/{i}_{j}.png')



def resize_and_convert(img, size, resample):
    
    image = Image.fromarray(img)
    resize_img = np.array(image.resize((size, size), Image.BICUBIC))
    resize_img = (resize_img - resize_img.min()) / (resize_img.max() - resize_img.min())

    return resize_img



def resize_multiple(img, sizes=(16, 128, 8), resample=Image.BICUBIC, lmdb_save=False, test = False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    vlr_img = resize_and_convert(lr_img, sizes[2], resample)
    if test ==False:
        # use avg pooling to resize the image
        image_tensor = torch.tensor(vlr_img).unsqueeze(0).unsqueeze(0).float()
        sr_img = F.interpolate(image_tensor, size=(sizes[0], sizes[0]), mode='nearest')
    else:
        image_tensor = torch.tensor(lr_img).unsqueeze(0).unsqueeze(0).float()
        sr_img = F.interpolate(image_tensor, size=(sizes[1], sizes[1]), mode='nearest')
    sr_img = np.array(sr_img[0][0])

    return [lr_img, hr_img, sr_img, vlr_img]


def resize_vlr(img, sizes, lmdb_save, test):
    # img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=Image.BICUBIC, lmdb_save=lmdb_save, test =test)
    lr_img, hr_img, sr_img, vlr_img = out
    return lr_img, hr_img, sr_img, vlr_img


#load the image
image = np.load(f'../../../dataset/fastMRI/multicoil_train/T2_image/0_0.npy')

# import pdb;pdb.set_trace()
plt.imshow(image, cmap='gray')
plt.savefig(f'../../../dataset/fastMRI/multicoil_train/T2_image/0_0.png')

# image = Image.open(f'../../../dataset/fastMRI/multicoil_train/T2_image/0_0.png')
# image = np.array(image)
lr_img, hr_img, sr_img, vlr_img = resize_vlr(image, sizes=( 128, 256, 64), lmdb_save=False, test=False)



plt.imshow(vlr_img, cmap='gray')
plt.savefig(f'../../../dataset/fastMRI/multicoil_train/T2_image/vlr_0_0.png')
plt.imshow(sr_img, cmap='gray')
plt.savefig(f'../../../dataset/fastMRI/multicoil_train/T2_image/sr_0_0.png')
plt.imshow(lr_img, cmap='gray')
plt.savefig(f'../../../dataset/fastMRI/multicoil_train/T2_image/lr_0_0.png')

import pdb; pdb.set_trace()



# class FastMRIDataset(Dataset):
#     def __init__(self, file_paths, transform=None):
#         """
#         Args:
#             file_paths (list of str): List of filenames for the HDF5 files containing fastMRI data.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.file_paths = file_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         with h5py.File(file_path, 'r') as file:
#             # Assuming the data is stored under the 'kspace' key. This will vary based on your data structure.
#             # You might need 'reconstruction_esc' or another key based on the dataset specifics.
#             import pdb; pdb.set_trace()
#             image = file['kspace'][()]
        
#         import pdb; pdb.set_trace()
#         # Convert to absolute value to collapse complex dimension
#         image = torch.from_numpy(np.abs(image))
        
#         if self.transform:
#             image = self.transform(image)

#         return image

# # Example transforms: Resize and convert to tensor. Adjust as needed.
# transforms = Compose([
#     ToTensor(),
#     Resize((128, 128))  # Resize to desired size for super resolution
# ])


# def create_dataloaders(file_paths, batch_size=4, train_split=0.8, transform=None):
#     # Split file_paths into training and validation sets
#     ##the file_path is a folder, and we need to get the file names
    
#     file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.h5')]


#     split_idx = int(len(file_paths) * train_split)
#     train_files = file_paths[:split_idx]
#     val_files = file_paths[split_idx:]
    
#     # Create dataset objects
#     train_dataset = FastMRIDataset(train_files, transform=transform)
#     val_dataset = FastMRIDataset(val_files, transform=transform)
    
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader

# # Usage
# file_paths = '../../../dataset/fastMRI/multicoil_train/'  # Add your file paths here
# train_loader, val_loader = create_dataloaders(file_paths, batch_size=4, transform=transforms)

# for images in train_loader:
#     print(images.shape)  # Check the output shape and tensor

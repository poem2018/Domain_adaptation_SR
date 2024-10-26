from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util

import numpy as np
import math
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torchvision.transforms.functional as TF
import core.metrics as Metrics

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        res_data = {}
        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # print(self.hr_path, self.sr_path, index)
            ##load npy file
            if 'npy' in self.hr_path[0]:
                if 'IXI' in self.hr_path[0]:
                    img_HR = np.load(self.hr_path[index])
                    img_SR = np.load(self.sr_path[index])
                    img_LR = np.load(self.lr_path[index])
                    img_HR = np.stack((img_HR,)*3, axis=-1)
                    img_SR = np.stack((img_SR,)*3, axis=-1)
                    img_LR = np.stack((img_LR,)*3, axis=-1)
                else:
                    img_HR = np.load(self.hr_path[index])
                    img_SR = np.load(self.sr_path[index])
                    img_LR = np.load(self.lr_path[index])
            else:
                img_HR = Image.open(self.hr_path[index]).convert("RGB")
                img_SR = Image.open(self.sr_path[index]).convert("RGB")
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
            # crop_area1 = (200,200,712,712)
            # img_HR = img_HR.crop(crop_area1)
            # img_SR = img_SR.crop(crop_area1)
            # print(np.shape(img_HR))
                
        scale = np.shape(img_HR)[0]/np.shape(img_LR)[0]
        
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            res_data =  {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index, 'scale': scale}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            res_data = { 'HR': img_HR, 'SR': img_SR, 'Index': index, 'scale': scale}


        # print("!!!", self.need_LR, index, res_data['HR'].shape)
        # import pdb;pdb.set_trace()
        # import torchvision.transforms.v2.functional as TF


        # img_HR = np.transpose(img_HR, (1, 2, 0))  # HWC, RGB
        # print("!!!",img_HR)
        # tensor1 = res_data['HR'] 
        # tensor2 = img_SR  #res_data['SR'] 
        # vis1 = Metrics.tensor2img(img_HR)
        # Metrics.save_img(vis1, './dataset/test/test1.png')

        # tensor2.save('./dataset/test/test2.png', format='PNG')
        
        
        return res_data


# import nibabel as nib

# # Path to your NIfTI file
# nifti_file_path = '../../dataset/IXI_T2/IXI002-Guys-0828-T2.nii'

# # Load the NIfTI file
# nifti_image = nib.load(nifti_file_path)

# # Get the data from the NIfTI object
# image_data = nifti_image.get_fdata()
# import pdb; pdb.set_trace()

# def load_IXI():

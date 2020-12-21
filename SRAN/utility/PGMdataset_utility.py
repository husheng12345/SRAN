import os
import glob
import numpy as np
from scipy import misc

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class PGMdataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*.npz")) if dataset_type in os.path.basename(f)]
        self.img_size = img_size 
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        meta_target = data["meta_target"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = np.arange(8)
            np.random.shuffle(indices)
            new_target = np.where(indices == target)[0][0]  
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target            
        
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)

        if meta_target.dtype == np.int8:
            meta_target = meta_target.astype(np.uint8)      
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)           
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target) 

        return resize_image, target, meta_target
        

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
import glob


def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info

        
def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    image_names=[]
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        elif key=="image_name":
            collated[key]=[ item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated



class RoadDataset(Dataset):
    def __init__(self, config, is_train, type_is="train"):
        self.is_train=is_train
        self.config = config
        assert self.config.DATASET in {'massachusetts', 'CHN6_CUG'}
        if self.config.DATASET == 'massachusetts':
            self.IMAGE_SIZE = 1500

            self.SAMPLE_MARGIN = 0
            self.image_paths = sorted(glob.glob(os.path.join(f"./massachusetts/{type_is}/image", "*.tiff")), key=lambda x: (
            int(os.path.basename(x).split("_")[0]), int(os.path.basename(x).split("_")[1].split(".")[0])))
            self.mask_paths= sorted(glob.glob(os.path.join(f"./massachusetts/{type_is}/mask", "*.tif")), key=lambda x: (
            int(os.path.basename(x).split("_")[0]), int(os.path.basename(x).split("_")[1].split(".")[0])))

        elif self.config.DATASET == 'CHN6_CUG':
            self.IMAGE_SIZE = 512
            self.SAMPLE_MARGIN = 0
            self.image_paths = sorted(glob.glob(os.path.join(f"./CHN6_CUG/{type_is}/image", "*.jpg")),
                                      key=lambda x: (
                                          int(os.path.basename(x).split("_")[0][2:])))    #3608
            self.mask_paths = sorted(glob.glob(os.path.join(f"./CHN6_CUG/{type_is}/mask", "*.png")),
                                     key=lambda x: (
                                         int(os.path.basename(x).split("_")[0][2:]))) #3608

        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)
        if self.config.LITTLE_DATA_TEST_CODE:
            self.image_paths=self.image_paths[:8]
            self.mask_paths=self.mask_paths [:8]
        if not self.is_train:#
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE)
            self.eval_patches = []
            for i in range(len(self.image_paths )):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                )
        a=1

    def __len__(self):
        return len(self.image_paths )

    def __getitem__(self, idx):

        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.image_paths))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:

            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
        if self.config.DATASET == 'massachusetts':
            #
            image=read_rgb_img(self.image_paths[img_idx])
            rgb_patch = image[begin_y:end_y, begin_x:end_x]
            mask=cv2.imread(self.mask_paths[img_idx],cv2.IMREAD_GRAYSCALE)
            mask_patch=mask[begin_y:end_y, begin_x:end_x][...,None]
        elif self.config.DATASET == 'CHN6_CUG':
            mask=cv2.imread(self.mask_paths[img_idx],cv2.IMREAD_GRAYSCALE)
            mask_patch=mask[begin_y:end_y, begin_x:end_x][...,None]
            path = self.mask_paths[img_idx].replace("_mask", "_sat").replace("mask", "image").replace("png", "jpg")
            image=cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_patch = image[begin_y:end_y, begin_x:end_x]

        if self.is_train:
            rot_index = np.random.randint(0, 4)
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            mask_patch = np.rot90(mask_patch, rot_index, [0,1]).copy()
        if self.config.DATASET == 'massachusetts':
            image_name = os.path.basename(self.mask_paths[img_idx]).split(".tif")[0]
        elif self.config.DATASET == 'CHN6_CUG':
            image_name = os.path.basename(self.mask_paths[img_idx]).split("_mask")[0]

        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'road_surface': torch.tensor(mask_patch, dtype=torch.float32) / 255.0,
            'image_name':image_name
        }
    def _concat_images(self, image1, image2,image3):
        if image1 is not None and image2 is not None and image3 is not None:
            img = np.concatenate([image1, image2,image3], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img
    def _concat_images_IT(self, image1, image2,):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img




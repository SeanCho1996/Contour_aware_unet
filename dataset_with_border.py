from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from PIL import Image
import torchvision


def ImageFetch(img_folder, split_rate=0.9):
    img_train = []
    mask_train = []
    img_val = []
    mask_val = []

    image_folder = os.path.join(img_folder, "image")
    mask_folder = os.path.join(img_folder, "mask")

    img_list = os.listdir(image_folder)
    total_img_num = len(img_list)
    print(f'Total number of images: {total_img_num}')

    train_num = int(total_img_num * split_rate)
    for idx, image_name in tqdm(enumerate(img_list[:train_num]), total=train_num, desc="load train images......"):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        img_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)
    for idx, image_name in tqdm(enumerate(img_list[train_num:]), total=(total_img_num - train_num), desc="load val images......"):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        img_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return img_train, mask_train, img_val, mask_val



def trainImageFetch(train_folder):
    image_train = []
    mask_train = []

    # load images and masks from their folders
    images_folder = os.path.join(train_folder, "image")
    masks_folder = os.path.join(train_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load train images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)

    return image_train, mask_train


def valImageFetch(val_folder):
    image_val = []
    mask_val = []

    images_folder = os.path.join(val_folder, "image")
    masks_folder = os.path.join(val_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load validation images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return image_val, mask_val


def make_onehot(mask, ignore_labels):
    mask_copy = deepcopy(mask)
    h, w = mask.shape[:]

    num_ignore = len(ignore_labels)
    # num_classes = len(torch.Tensor.unique(mask)) - num_ignore
    num_classes = 2

    zeros = torch.zeros(((num_classes + num_ignore), h, w))

    # decrease ignore labels' indexes into successive integers(eg: convert 0, 1, 2, 255 into 0, 1, 2, 3)
    for i in range(num_ignore):
        mask_copy[mask_copy == ignore_labels[i]] = num_classes + i

    # scatter to one_hot
    one_hot = zeros.scatter_(1, mask_copy.unsqueeze(0), 1)

    return one_hot[:num_classes, :, :]


# class SegDataset(Dataset):
#     def __init__(self, image_list, mask_list, mode, transform_img, transform_mask, transform_border):
#         self.mode = mode
#         self.transform_img = transform_img
#         self.transform_mask = transform_mask
#         self.transform_border = transform_border
#         self.imagelist = image_list
#         self.masklist = mask_list


#     def __len__(self):
#         return len(self.imagelist)


#     def __getitem__(self, idx):
#         image = deepcopy(self.imagelist[idx])

#         if self.mode == 'train':
#             mask = deepcopy(self.masklist[idx])
            
#             mask_arr = np.array(mask)
#             border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
#             border /= 255
#             border = Image.fromarray(border.astype(np.uint8))
#             border_img = self.transform_border(border)
#             border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
#             border_oh = make_onehot(border, [255])
#             # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

#             image = self.transform_img(image)

#             mask = self.transform_mask(mask)
#             mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
#             # print(f'after transform mask max: {mask.max()}')

#             # image = image.unsqueeze(0)
#             # mask = mask.unsqueeze(0)

#             return image, mask, border, border_oh

#         elif self.mode == 'val':
#             mask = deepcopy(self.masklist[idx])

#             mask_arr = np.array(mask)
#             border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
#             border /= 255
#             border = Image.fromarray(border.astype(np.uint8))
#             border_img = self.transform_border(border)
#             border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
#             border_oh = make_onehot(border, [255])
#             # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

#             image = self.transform_img(image)

#             mask = self.transform_mask(mask)
#             mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

#             # image = image.unsqueeze(0)
#             # mask = mask.unsqueeze(0)

#             return image, mask, border, border_oh

class SegDataset(Dataset):
    def __init__(self, image_list, mask_list, mode, transform_img, transform_mask, transform_border):
        self.mode = mode
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_border = transform_border
        self.imagelist = image_list
        self.masklist = mask_list


    def __len__(self):
        return len(self.imagelist)


    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            
            mask_arr = np.array(mask)
            border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
            border /= 255
            border = Image.fromarray(border.astype(np.uint8))
            border_img = self.transform_border(border)
            border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
            # border_oh = make_onehot(border, [255])
            # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
            # print(f'after transform mask max: {mask.max()}')

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask, border

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            mask_arr = np.array(mask)
            border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
            border /= 255
            border = Image.fromarray(border.astype(np.uint8))
            border_img = self.transform_border(border)
            border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
            # border_oh = make_onehot(border, [255])
            # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask, border


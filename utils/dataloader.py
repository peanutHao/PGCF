import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# class Data(data.Dataset):
#     def __init__(self, image_root, gt_root):
#         self.image_root = image_root
#         self.gt_root = gt_root
#         self.samples   = [name for name in os.listdir(image_root) if name[0]!="."]
#         # self.transform = A.Compose([
#         #     A.Normalize(),
#         #     A.Resize(352, 352),
#         #     A.HorizontalFlip(p=0.5),
#         #     A.VerticalFlip(p=0.5),
#         #     A.RandomRotate90(p=0.5),
#         #     ToTensorV2()
#         # ])
#         self.img_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((352, 352)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])])
#
#         self.gt_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((352, 352)),
#             transforms.ToTensor()])
#         # transforms.Compose([
#         #     transforms.Resize((self.trainsize, self.trainsize)),
#         #     transforms.ToTensor()])
#         self.color1, self.color2 = [], []
#         for name in self.samples:
#             if name[:-4].isdigit():
#                 self.color1.append(name)
#             else:
#                 self.color2.append(name)
#
#     def __getitem__(self, idx):
#         name  = self.samples[idx]
#         image = cv2.imread(self.image_root+name)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#
#         name2  = self.color1[idx%len(self.color1)] if np.random.rand()<0.7 else self.color2[idx%len(self.color2)]
#         image2 = cv2.imread(self.image_root+name2)
#         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
#
#         mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
#         mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
#         image = np.uint8((image-mean)/std*std2+mean2)
#         image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
#         mask  = cv2.imread(self.gt_root+name, cv2.IMREAD_GRAYSCALE)
#                 # /255.0
#         # pair  = self.transform(image=image, mask=mask)
#         # print(mask)
#         # mask = self.gt_transform(mask.astype(np.uint8))
#         # image = self.img_transform(image.astype(np.uint8))
#         mask = self.gt_transform(mask)
#         image = self.img_transform(image)
#         # print(mask)
#         # return pair['image'], pair['mask']
#         return image, mask
#     def __len__(self):
#         return len(self.samples)
class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # print("zhiqian")
        # print(gt.size)
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        # seed = 2222
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        # print("zhihou")
        # print(gt.shape)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    # dataset = Data(image_root, gt_root)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

from dataset_utils import get_img, resize_bbox, multiple_resize_bbox, bias_multiple_resize_bbox, create_mask, create_maskExtended


class Bbox:
    def __init__(self, x1, x2, y1, y2, cls):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cls = cls
    
    def __str__(self):
        return f"X1: {self.x1} Y1: {self.y1} X2: {self.x2} Y2: {self.y2} cls: {self.cls}"

class ObjDectDS(Dataset):

    #TODO:
    #   - mosaic (random cut area)
    #   - cutout
    def __init__(self, dataset_path,
                        trainval, 
                        num_classes, 
                        mosaic_prob = 0.2, 
                        mixup_prob = 0.15, 
                        colorJitter_prob = .15, 
                        mosaic_colorJitter_prob = 0.15, 
                        mixup_colorJitter_prob = 0.1,
                        gauss_noise_prob = 0.15,
                        mosaic_gauss_noise_prob = 0.15,
                        mixup_gauss_noise_prob = 0.1, 
                        small_size = 48, 
                        medium_size = 96,
                        large_size = 512,
                        extended = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.trainval = trainval
        self.get_names()
        self.toTensor = transforms.ToTensor()
        self.num_classes = num_classes
        self.small_size = small_size
        self.medium_size = medium_size
        self.large_size = large_size
        self.extended = extended


        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.colorJitter_prob = colorJitter_prob
        self.mosaic_colorJitter_prob = mosaic_colorJitter_prob
        self.mixup_colorJitter_prob = mixup_colorJitter_prob
        self.gauss_noise_prob = gauss_noise_prob
        self.mosaic_gauss_noise_prob = mosaic_gauss_noise_prob
        self.mixup_gauss_noise_prob = mixup_gauss_noise_prob

        self.aug_CLAHE = A.CLAHE(always_apply=True)
        self.aug_ColorJitter = A.ColorJitter(always_apply=True)
        self.aug_Equalize = A.Equalize(always_apply=True)
        self.aug_GaussNoise = A.GaussNoise(var_limit=(10, 200), always_apply=True)

        #self.cls_names_a = ['0', '1', '2', '3', '4', '6', '7', '8', '9']
        #self.cls2idx = {'0':0, '1':1, '2':2, '3':3, '4':4, '6':5, '7':6, '8':7, '9':8}

        self.cls_names_a = ['2', '3', '4']
        self.cls2idx = {'2':0, '3':1, '4':2}


    def get_names(self):
        self.names = []
        for folder in os.listdir(os.path.join(self.dataset_path, self.trainval)):
            folder_imgs = os.listdir(os.path.join(self.dataset_path, self.trainval, folder, 'annotations'))
            for imgs in folder_imgs:
                self.names.append(os.path.join(self.dataset_path, self.trainval, folder, 'annotations',imgs))    
        self.names = list(map(lambda s: s.replace('.txt',''), self.names))
        
    
    def augment_image(self, img, idx):
        im = img
        if idx == 0:
            im = self.aug_ColorJitter(image = im)['image']
        elif idx == 2:
            im = self.aug_GaussNoise(image = im)['image'] 
        return im
        
    def mixup(self, img_path, anno_path, new_image_shape):
        img1 = get_img(img_path)
        ####img1#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mixup_colorJitter_prob, self.mixup_gauss_noise_prob, 1.-(self.mixup_colorJitter_prob +  self.mixup_gauss_noise_prob)])
        img1 = self.augment_image(img1, aug_img_idx)

        #get second image, anno path
        idxs = torch.randint(0, len(self.names), (1,))
        img_path2 = os.path.join(f"{self.names[idxs[0]].replace('annotations', 'new_images')}.jpg")
        anno_path2 = os.path.join(f"{self.names[idxs[0]]}.txt")

        img2 = get_img(img_path2)
        ####img2#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mixup_colorJitter_prob, self.mixup_gauss_noise_prob, 1.-(self.mixup_colorJitter_prob +  self.mixup_gauss_noise_prob)])
        img2 = self.augment_image(img2, aug_img_idx)

        old_sizes = [(img1.shape[1], img1.shape[0]), (img2.shape[1], img2.shape[0])]
        img1 = cv2.resize(img1, (new_image_shape[0], new_image_shape[1]))
        img2 = cv2.resize(img2, (new_image_shape[0], new_image_shape[1]))
        final_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        final_img_tensor = self.toTensor(final_img)

        anno_pathes = [anno_path, anno_path2]

        tabBbox = multiple_resize_bbox(anno_pathes, old_sizes, new_image_shape, self.cls_names_a, self.cls2idx)
        tabBbox.sort(key=lambda x: (x.x2-x.x1)*(x.y2-x.y1), reverse=True)
        if self.extended:
            labels = create_maskExtended(tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size, self.large_size)
        else:
            labels = create_mask(tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size)
        

        return (final_img_tensor, labels)
    
    def basic_mask(self, img_path, anno_path, new_image_shape):
        #print(img_path)
        img = get_img(img_path)
        ###img### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.colorJitter_prob, self.gauss_noise_prob, 1.-(self.colorJitter_prob +  self.gauss_noise_prob)])
        img = self.augment_image(img, aug_img_idx)
        img_shape = (img.shape[1], img.shape[0])
        resized_img = cv2.resize(img, new_image_shape)
        final_img_tensor =  self.toTensor(resized_img)

        tabBbox = resize_bbox(anno_path, img_shape, new_image_shape,  self.cls_names_a, self.cls2idx)
        tabBbox.sort(key=lambda x: (x.x2-x.x1)*(x.y2-x.y1), reverse=True)
        if self.extended:
            labels = create_maskExtended(tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size, self.large_size)
        else:
            labels = create_mask(tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size)

        return (final_img_tensor, labels)
    
    def mosaic(self, img_path, anno_path, new_image_shape):
        img1 = get_img(img_path)
        ####img1#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mosaic_colorJitter_prob, self.mosaic_gauss_noise_prob, 1.-(self.mosaic_colorJitter_prob +  self.mosaic_gauss_noise_prob)])
        img1 = self.augment_image(img1, aug_img_idx)

        idxs = torch.randint(0, len(self.names), (3,))
        img_path2 = os.path.join(f"{self.names[idxs[0]].replace('annotations', 'new_images')}.jpg")
        anno_path2 = os.path.join(f"{self.names[idxs[0]]}.txt")
        img_path3 = os.path.join(f"{self.names[idxs[1]].replace('annotations', 'new_images')}.jpg")
        anno_path3 = os.path.join(f"{self.names[idxs[1]]}.txt")
        img_path4 = os.path.join(f"{self.names[idxs[2]].replace('annotations', 'new_images')}.jpg")
        anno_path4 = os.path.join(f"{self.names[idxs[2]]}.txt")
        ##### ZMIANA KODU
        #print(img_path, img_path2, img_path3, img_path4)
        img2 = get_img(img_path2)
        ####img2#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mixup_colorJitter_prob, self.mixup_gauss_noise_prob, 1.-(self.mixup_colorJitter_prob +  self.mixup_gauss_noise_prob)])
        img2 = self.augment_image(img2, aug_img_idx)

        img3 = get_img(img_path3)
        ####img3#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mixup_colorJitter_prob, self.mixup_gauss_noise_prob, 1.-(self.mixup_colorJitter_prob +  self.mixup_gauss_noise_prob)])
        img3 = self.augment_image(img3, aug_img_idx)

        img4 = get_img(img_path4)
        ####img4#### augmentation - color jitter
        aug_img_idx = np.random.choice(3, 1, p=[self.mixup_colorJitter_prob, self.mixup_gauss_noise_prob, 1.-(self.mixup_colorJitter_prob +  self.mixup_gauss_noise_prob)])
        img4 = self.augment_image(img4, aug_img_idx)


        old_sizes = [(img1.shape[1], img1.shape[0]), (img2.shape[1], img2.shape[0]), (img3.shape[1], img3.shape[0]), (img4.shape[1], img4.shape[0])]


        anno_pathes = [anno_path, anno_path2, anno_path3, anno_path4]
        bias = [(0,0), (new_image_shape[0], 0), (0, new_image_shape[1]), (new_image_shape[0], new_image_shape[1])]

        tabBbox = bias_multiple_resize_bbox(anno_pathes, old_sizes, new_image_shape, bias,  self.cls_names_a, self.cls2idx)
        tabBbox.sort(key=lambda x: (x.x2-x.x1)*(x.y2-x.y1), reverse=True)

        x_rand = torch.randint(int(new_image_shape[0]*0.3), int(new_image_shape[0]*0.9), (1,)).item()
        y_rand = torch.randint(int(new_image_shape[1]*0.1), int(new_image_shape[1]*0.9), (1,)).item()

        new_x1 = x_rand
        new_y1 = y_rand
        new_x2 = new_x1 + new_image_shape[0] 
        new_y2 = new_y1 + new_image_shape[1]

        croped_tabBbox = []
        for tab in tabBbox:
            x1_intersection = max(tab.x1, new_x1)
            x2_intersection = min(tab.x2, new_x2)
            y1_intersection = max(tab.y1, new_y1)
            y2_intersection = min(tab.y2, new_y2)

            intersection = (x2_intersection - x1_intersection)*(y2_intersection - y1_intersection)
            if intersection > 0:
                new_bbox_x1 = x1_intersection - new_x1
                new_bbox_x2 = x2_intersection - new_x1
                new_bbox_y1 = y1_intersection - new_y1
                new_bbox_y2 = y2_intersection - new_y1

                croped_tabBbox.append(Bbox(new_bbox_x1, new_bbox_x2, new_bbox_y1, new_bbox_y2, tab.cls))
        croped_tabBbox.sort(key=lambda x: (x.x2-x.x1)*(x.y2-x.y1), reverse=True)
        if self.extended:
            labels = create_maskExtended(croped_tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size, self.large_size)
        else:
            labels = create_mask(croped_tabBbox, new_image_shape, self.num_classes, self.small_size, self.medium_size)
        #tutaj dodaj obrot obrazka

        img1 = cv2.resize(img1, (new_image_shape))
        img2 = cv2.resize(img2, (new_image_shape))
        img3 = cv2.resize(img3, (new_image_shape))
        img4 = cv2.resize(img4, (new_image_shape))

        img12 = cv2.hconcat([img1, img2])
        img34 = cv2.hconcat([img3, img4])
        final_img = cv2.vconcat([img12, img34])
        final_img = final_img[new_y1:new_y2, new_x1:new_x2, :]
        final_img = self.toTensor(final_img)

        return (final_img, labels)
    
    def __getitem__(self, batch):
        idx, new_image_shape = batch
        name = self.names[idx]
        img_path = os.path.join(f"{name.replace('annotations', 'new_images')}.jpg")
        anno_path = os.path.join(f"{name}.txt")
        aug_idx = np.random.choice(3, 1, p=[self.mosaic_prob, self.mixup_prob, 1-(self.mosaic_prob+self.mixup_prob)])
        if aug_idx == 0:
            return self.mosaic(img_path, anno_path, new_image_shape)
        elif aug_idx == 1:
            return self.mixup(img_path, anno_path, new_image_shape)
        elif aug_idx == 2:
            return self.basic_mask(img_path, anno_path, new_image_shape)
        
    def __len__(self):
        return len(self.names)








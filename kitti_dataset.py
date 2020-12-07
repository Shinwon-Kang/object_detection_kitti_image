from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

import random

class KittiDataset(torch.utils.data.Dataset):
    
    def __init__(self, is_train, root_dir):
        super().__init__()
        
        self.is_train = is_train
        self.root_dir = root_dir

        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "image_2"))))
        self.labels = list(sorted(os.listdir(os.path.join(root_dir, "label_2"))))

        self.label_encoder = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, 
                              "Person_sitting": 4, "Cyclist": 5, "Tram": 6, 
                              "Misc": 7, "DontCare": 8}

        # self.training_image_dir = root_dir + "image_2/"
        # self.training_label_dir = root_dir + "label_2/"
        # self.training_velodyne_dir = root_dir + "velodyne/"        
        # self.training_calib_dir = root_dir + "calib/"

        if self.is_train:
            self.transform = Transform(train=True)
        else:
            self.transform = Transform(train=False)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "image_2", self.imgs[idx])
        label_path = os.path.join(self.root_dir, "label_2", self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image).astype('float32') / 255.0
        # image /= 255.0

        # object name, bounding box (x, y, w, h)
        labels, boxes, areas = self.get_objects_data(label_path, [1242, 375])

        # resize, flip, sharpness ...
        # image, boxes = self.transform.apply_transform(image, boxes)

        # area
        # boxes = np.array(boxes)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # to Tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # image Normalize
        image = torchvision.transforms.ToTensor()(image)
        # image = self.transform.normalize(self.transform.to_tensor(image))

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = areas

        return image, target

    def get_objects_data(self, label_path, size):
        labels = []
        boxes = []
        areas = []
        with open(label_path, "r") as f:
            for line in f:
                labels_data = line.split()

                # size : 1242, 375
                # labels.append(self.label_encoder[labels_data[0]])
                label = self.label_encoder[labels_data[0]]
                if label is not 'DontCare':
                    labels.append(label)
                    # boxes.append([float(labels_data[4]) / size[0], 
                    #               float(labels_data[5]) / size[1], 
                    #               float(labels_data[6]) / size[0], 
                    #               float(labels_data[7]) / size[1]])

                    # x_min, x_max = float(labels_data[4]) / size[0], float(labels_data[6]) / size[0]
                    # y_min, y_max = float(labels_data[5]) / size[1], float(labels_data[7]) / size[1]

                    # boxes.append([float(labels_data[4]), float(labels_data[5]), 
                    #                         float(labels_data[6]) - float(labels_data[4]), 
                    #                         float(labels_data[7]) - float(labels_data[5])])
                    
                    boxes.append([float(labels_data[4]), float(labels_data[5]), 
                                    float(labels_data[6]), float(labels_data[7])])

                    areas.append((float(labels_data[7]) - float(labels_data[5])) 
                                * (float(labels_data[6]) - float(labels_data[4])))

                    # Return c_x, c_y, w, h
                    # boxes.append([(x_min + x_max) / 2, 
                    #                (y_min + y_max) / 2, 
                    #                x_max - x_min, 
                    #                y_max - y_min])
        
        return labels, boxes, areas



class Transform(object):
    """docstring for Transform"""
    def __init__(self,  train):
        super(Transform, self).__init__()
        self.train = train 
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def apply_transform(self, image, boxes) :
        if self.train : 
            if random.random() < 0.5:
                image , boxes = self.flip(image, boxes)
          
            if random.random() < 0.5:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1/8)
            
            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(factor)

            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(factor)

            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(factor)
                
        return image , boxes
    
    # dataformat : ith index ==> img_data == dict, keys : 'bboxes' , 'image' , 'class', len(dict[bboxes] == len(dict[bboxes])
    def flip(self, image, boxes):
        # Flip image
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # new_image = FT.hflip(image)
        # Flip boxes
        boxes = [ [image.width - cord -1 if i % 2 ==0 else cord for i,cord in enumerate(box)  ] for box in boxes]
        boxes = [ [box[2] ,box[1] , box[0], box[3]] for box in boxes]
        return new_image, boxes
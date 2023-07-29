import os
import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from parameters import *
import matplotlib.pyplot as plt

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        super(TripletFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.person_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

    def __len__(self) -> int:
        return len(self.person_folders)
    
    def _get_images_from_folder(self, folderPath):
        images = [img_name for img_name in os.listdir(folderPath) if img_name.endswith('.jpg')]
        return images
    
    def _random_triplet_indices(self, numImages):
        anchor_idx, positive_idx = torch.randperm(numImages)[:2]
        return anchor_idx, positive_idx
        
    def __getitem__(self, index):
        person_folder = self.person_folders[index]
        person_path = os.path.join(self.root_dir, person_folder)
        images = self._get_images_from_folder(person_path)
        num_images = len(images)

        if num_images < 2:
            dummy_image = Image.new('RGB', (96, 96))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, dummy_image, dummy_image
        
        anchor_idx, positive_idx = self._random_triplet_indices(num_images)
        anchor_path = os.path.join(person_path, images[anchor_idx])
        positive_path = os.path.join(person_path, images[positive_idx])

        negative_people = [self.person_folders[i] for i in range(len(self.person_folders)) if i != index and len(self._get_images_from_folder(os.path.join(self.root_dir, self.person_folders[i]))) >= 1]
        negative_person = random.choice(negative_people)
        negative_path = os.path.join(self.root_dir, negative_person, random.choice(os.listdir(os.path.join(self.root_dir, negative_person))))

        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_image = Image.open(positive_path).convert("RGB")
        negative_image = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        
        return anchor_image, positive_image, negative_image
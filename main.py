import train
import test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from PIL import Image
from parameters import *
from loss import *
from data import *

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return TripletFaceDataset(root_dir= r"cropped", transform= transform)

def split_dataset(dataset):
    val_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(val_split * dataset_size)
    train_indices, val_indices = indices[split : ], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size= BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    return train_loader, val_loader

def main():
    dataset = load_dataset()
    train_loader, test_loader = split_dataset(dataset)
    train.train(train_loader)
    test.test(test_loader)

if __name__ == '__main__':
    main()
    
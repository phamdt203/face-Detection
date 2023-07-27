import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from parameters import *
from loss import *


def main():
    num_classes = 1000
    model = InceptionResnetV1()
    model = model.to('cuda')
    loss = tripletLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    num_epochs = NUM_EPOCHS
    train_model = (model, loss, )
import train
import test
import torch.optim as optim
from model import *
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from parameters import *
from data import *
from utilsData import load_database
from model import *
import torch.nn as nn

def load_model():
    model = mobilenet_v2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def load_dataset(transform):
    return TripletFaceDataset(root_dir= r"cropped", transform= transform)

def split_dataset(dataset):
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    num_samples = len(dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = int(val_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples - num_val_samples

    train_dataset, temp_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    val_dataset, test_dataset = random_split(temp_dataset, [num_val_samples, num_test_samples])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.TripletMarginLoss(margin = ALPHA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    dataset = load_dataset(preprocess)
    train_loader, val_loader, test_loader = split_dataset(dataset)
    train.train(train_loader, val_loader, model, device, optimizer, loss_fn, NUM_EPOCHS)
    print(f"Accuracy :  {test.test(test_loader, model, device)}")
    database, paths = load_database(model, preprocess, device)

if __name__ == '__main__':
    main()
    
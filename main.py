import train
import test
import torch.optim as optim
from model import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from PIL import Image
from predict import *
from parameters import *
from data import *
from utilsData import load_database

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(pretrained=False)
    model = model.to(device)
    return model

def load_dataset(transform):
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
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.TripletMarginLoss(margin = ALPHA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = load_dataset(preprocess)
    train_loader, test_loader = split_dataset(dataset)
    train.train(train_loader, model, device, optimizer, loss_fn, NUM_EPOCHS)
    print(f"Accuracy :  {test.test(test_loader, model, device)}")
    database, paths = load_database(model, preprocess, device)
    faceRecognition(database, paths, device, model, preprocess)

if __name__ == '__main__':
    main()
    
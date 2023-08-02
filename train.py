import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from parameters import *
from loss import *
from data import *

def train_model(model, train_loader, loss_fn, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for i, triplet in enumerate(train_loader):
            anchors, positives, negatives = triplet
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)
            loss = loss_fn(anchors_embedding, positives_embedding, negatives_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1} / {num_epochs}], image [{i + 1} / {len(train_loader)}], loss value : {loss.item()}")
    torch.save(model.state_dict(), "facenet_model.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
    num_epochs = NUM_EPOCHS
    # transform = transforms.Compose([
    #     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    # ])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    path = r"cropped"
    dataset = TripletFaceDataset(root_dir= path, transform= transform)
    loss_fn = nn.TripletMarginLoss(margin = ALPHA)
    dataset = DataLoader(dataset = dataset, batch_size= BATCH_SIZE)
    # Training 
    train_model(model = model, train_loader = dataset, loss_fn = loss_fn, optimizer= optimizer, num_epochs= num_epochs, device = device)  

if __name__ == '__main__':
    main()
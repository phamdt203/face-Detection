import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from parameters import *
from loss import *
from data import *

def train_model(model, train_loader, loss_fn, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for triplet in train_loader:
            anchors, positives, negatives = triplet
            anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)
            loss = loss_fn(anchors_embedding, positives_embedding, negatives_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

def main():
    model = InceptionResnetV1()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
    num_epochs = NUM_EPOCHS
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])


    path = r"C:\Users\Dell\OneDrive\Desktop\Code\AI\AI\faceRecognition\cropped"
    dataset = TripletFaceDataset(root_dir= path, transform= transform)
    num_samples = len(dataset)
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    loss_fn = nn.TripletMarginLoss(margin = ALPHA)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE)
    # valid_loader = DataLoader(valid_set, batch_size= BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(dataset = test_set, batch_size=BATCH_SIZE)

    # Training 
    train_model(model = model, train_loader = train_loader, loss_fn = loss_fn, optimizer= optimizer, num_epochs= num_epochs)

if __name__ == '__main__':
    main()
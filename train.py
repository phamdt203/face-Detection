import torch
import torch.nn as nn
import torch.optim as optim
from parameters import *
from loss import *
from data import *

def train(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    num_epochs = NUM_EPOCHS
    loss_fn = nn.TripletMarginLoss(margin = ALPHA)
    model.train(True)
    for epoch in range(num_epochs):
        for i, triplet in enumerate(train_loader):
            anchors, positives, negatives = triplet
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)
            loss = loss_fn(anchors_embedding, positives_embedding, negatives_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1} / {num_epochs}], image [{i + 1} / {len(train_loader)}], loss value : {loss.item()}")
    torch.save(model.state_dict(), "facenet_model.pth")
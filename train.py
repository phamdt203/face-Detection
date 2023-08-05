import torch
import torch.nn as nn
from parameters import *

def train(train_loader, model, device, optimizer, loss_fn, num_epochs):
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
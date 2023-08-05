import torch
import torch.nn as nn
from parameters import *

def evalute(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(train_loader, val_loader, model, device, optimizer, loss_fn, num_epochs):
    model.train(True)
    for epoch in range(num_epochs):
        loss_total = 0
        for i, triplet in enumerate(train_loader[:5]):
            anchors, positives, negatives = triplet
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)
            loss = loss_fn(anchors_embedding, positives_embedding, negatives_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f"Epoch [{epoch + 1} / {num_epochs}], average loss value : {loss_total / len(train_loader)}")
    print(f"Accuracy : {evalute(model, val_loader)}")
    torch.save(model.state_dict(), "facenet_model.pth")
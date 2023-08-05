import torch
import torch.nn as nn
from parameters import *
import torch.nn.functional as F

def evalute(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    for triplet in val_loader:
        anchors, positives, negatives = triplet
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)

        positives_distance = F.pairwise_distance(anchors_embedding, positives_embedding)
        negatives_distance = F.pairwise_distance(anchors_embedding, negatives_embedding)

        correct += torch.sum(positives_distance < negatives_distance).item()
        total += anchors_embedding.size(0)
    return (correct / total) * 100  

def train(train_loader, val_loader, model, device, optimizer, loss_fn, num_epochs):
    model.train(True)
    # number_of_batch = 5
    for epoch in range(num_epochs):
        loss_total = 0
        for i, triplet in enumerate(train_loader):
            anchors, positives, negatives = triplet
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_embedding, positives_embedding, negatives_embedding = model(anchors), model(positives), model(negatives)
            loss = loss_fn(anchors_embedding, positives_embedding, negatives_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            # if i == number_of_batch:
            #     break
        print(f"Epoch [{epoch + 1} / {num_epochs}], average loss value : {loss_total / len(train_loader)}")
    print(f"Accuracy : {evalute(model, val_loader, device)}")
    torch.save(model.state_dict(), "facenet_model.pth")
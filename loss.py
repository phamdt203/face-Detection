import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        
        losses = distance_positive - distance_negative + self.margin
        losses = torch.clamp(losses, min=0.0)
        
        triplet_loss = torch.mean(losses)
        
        return triplet_loss

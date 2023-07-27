import torch

class tripletLoss():
    def __init__(self, anchors, positives, negatives, alpha):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.alpha = alpha

    def total_triplet_loss(self):
        total_loss = 0.0
        for i in range(len(self.anchors)):
            pos_dist = torch.sum(torch.square(self.anchors[i] - self.positives[i]), dim=1)
            neg_dist = torch.sum(torch.square(self.anchors[i] - self.negatives[i]), dim=1)
    
            basic_loss = pos_dist - neg_dist + self.alpha
            loss = torch.mean(torch.max(basic_loss, torch.zeros_like(basic_loss)))
            total_loss += loss
        return total_loss / len(self.anchors)
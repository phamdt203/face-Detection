import torch
import torch.nn.functional as F

def get_embbeding(model, image):
    with torch.no_grad():
        embedding = model(image)
    return F.normalize(embedding, p = 2, dim =1)

def test(test_loader, model, device):
    model.load_state_dict(torch.load("facenet_model.pth"))
    model.eval()
    correct = 0
    total = 0
    for triplet in test_loader:
        anchors, positives, negatives = triplet
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        anchors_embedding, positives_embedding, negatives_embedding = get_embbeding(model, anchors), get_embbeding(model,positives), get_embbeding(model, negatives)

        positives_distance = F.pairwise_distance(anchors_embedding, positives_embedding)
        negatives_distance = F.pairwise_distance(anchors_embedding, negatives_embedding)

        correct += torch.sum(positives_distance < negatives_distance).item()
        total += anchors_embedding.size(0)
    return (correct / total) * 100  


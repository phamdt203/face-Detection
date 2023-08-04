import torch
import torch.nn.functional as F

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained = True)
    model = model.to(device)
    model.load_state_dict(torch.load("facenet_model.pth"))
    model.eval()
    return model

def get_embbeding(model, image):
    with torch.no_grad():
        embedding = model(image)
    return F.normalize(embedding, p = 2, dim =1)

def test(test_loader):
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    for triplet in test_loader:
        anchors, positives, negatives = triplet
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        anchors_embedding, positives_embedding, negatives_embedding = get_embbeding(anchors), get_embbeding(positives), get_embbeding(negatives)

        positives_distance = F.pairwise_distance(anchors_embedding, positives_embedding)
        negatives_distance = F.pairwise_distance(anchors_embedding, negatives_embedding)

        correct += torch.sum(positives_distance < negatives_distance).item()
        total += anchors_embedding.size(0)
    return (correct / total) * 100  


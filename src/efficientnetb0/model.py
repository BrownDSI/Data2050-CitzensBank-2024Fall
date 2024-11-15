import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, num_classes=2):
        super(TripletSignatureEmbeddingModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, embedding_size, num_classes)

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.efficientnet(anchor)
        positive_embedding = self.efficientnet(positive)
        negative_embedding = self.efficientnet(negative)
        return anchor_embedding, positive_embedding, negative_embedding

def get_embedding_model(num_classes=2):
    model = TripletSignatureEmbeddingModel(num_classes=num_classes)
    return model

# Custom loss function for triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
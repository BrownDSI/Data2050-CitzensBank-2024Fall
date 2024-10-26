import torch
import torch.nn as nn
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


import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_BN_Weights

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, num_classes=2):
        super(TripletSignatureEmbeddingModel, self).__init__()
        self.vgg16 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        num_ftrs = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier = nn.Sequential(
            *list(self.vgg16.classifier.children())[:-1],  # Remove the last layer
            nn.Linear(num_ftrs, embedding_size)  # Add a new layer with the embedding size
        )

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.vgg16(anchor)
        positive_embedding = self.vgg16(positive)
        negative_embedding = self.vgg16(negative)
        return anchor_embedding, positive_embedding, negative_embedding

def get_embedding_model(num_classes=2):
    model = TripletSignatureEmbeddingModel(num_classes=num_classes)
    return model

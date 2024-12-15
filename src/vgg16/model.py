# import torch
# import torch.nn as nn
# from torchvision import models
# from torchvision.models import VGG16_BN_Weights

# class TripletSignatureEmbeddingModel(nn.Module):
#     def __init__(self, embedding_size=256, num_classes=2):
#         super(TripletSignatureEmbeddingModel, self).__init__()
#         self.vgg16 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
#         num_ftrs = self.vgg16.classifier[-1].in_features
#         self.vgg16.classifier = nn.Sequential(
#             *list(self.vgg16.classifier.children())[:-1],  # Remove the last layer
#             nn.Linear(num_ftrs, embedding_size)  # Add a new layer with the embedding size
#         )

#     def forward(self, anchor, positive, negative):
#         anchor_embedding = self.vgg16(anchor)
#         positive_embedding = self.vgg16(positive)
#         negative_embedding = self.vgg16(negative)
#         return anchor_embedding, positive_embedding, negative_embedding

# def get_embedding_model(num_classes=2):
#     model = TripletSignatureEmbeddingModel(num_classes=num_classes)
#     return model

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_BN_Weights
import torch.nn.functional as F

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, freeze_features=True):
        super(TripletSignatureEmbeddingModel, self).__init__()
        self.vgg16 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    
        block_start_indices = [0, 5, 10, 17, 24]  # Indices for VGG16 blocks
        freeze_up_to_block = 2
        for i, child in enumerate(self.vgg16.features.children()):
            if i < block_start_indices[freeze_up_to_block]:
                for param in child.parameters():
                    param.requires_grad = False

        num_ftrs = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier = nn.Sequential(
            *list(self.vgg16.classifier.children())[:-1],  # Remove the last layer
            nn.Linear(num_ftrs, embedding_size),           # Add embedding layer
            nn.ReLU(),                                     # Add non-linearity
            nn.BatchNorm1d(embedding_size)                 # Normalize embeddings
        )

    def forward(self, anchor, positive, negative):
        combined = torch.cat([anchor, positive, negative], dim=0)
        embeddings = self.vgg16(combined)
        anchor_embedding, positive_embedding, negative_embedding = torch.split(embeddings, anchor.size(0))


        return anchor_embedding, positive_embedding, negative_embedding

def get_embedding_model(embedding_size=256, freeze_features=True):
    model = TripletSignatureEmbeddingModel(embedding_size=embedding_size, freeze_features=freeze_features)
    return model

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


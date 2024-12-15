import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model # import pretrainedmodels  # Community package for Xception

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, num_classes=2):
        super(TripletSignatureEmbeddingModel, self).__init__()
        # Load pretrained Xception model
        self.xception = create_model('xception', pretrained=True)
        num_ftrs = self.xception.get_classifier().in_features
        
        # Replace the classifier with an embedding layer
        self.xception.reset_classifier(num_classes=0)  # Remove original classifier
        self.embedding_layer = nn.Linear(num_ftrs, embedding_size)

    def forward(self, anchor, positive, negative):
        # Get embeddings
        anchor_embedding = self.embedding_layer(self.xception(anchor))
        positive_embedding = self.embedding_layer(self.xception(positive))
        negative_embedding = self.embedding_layer(self.xception(negative))
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
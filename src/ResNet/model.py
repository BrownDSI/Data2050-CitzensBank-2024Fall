import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetModel

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, num_classes=2):
        super(TripletSignatureEmbeddingModel, self).__init__()
        # Load the pretrained ResNet model from Hugging Face Transformers
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50')
        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, num_features, 1, 1)
        # Replace the final fully connected layer
        num_ftrs = self.resnet.config.hidden_sizes[-1]  # Get the number of output features
        self.embedding_layer = nn.Linear(num_ftrs, embedding_size)

    def forward(self, anchor, positive, negative):
        # Extract features for each input
        anchor_features = self.resnet(anchor).last_hidden_state  # Shape: (batch_size, num_features, h, w)
        positive_features = self.resnet(positive).last_hidden_state
        negative_features = self.resnet(negative).last_hidden_state

        # Apply global average pooling to reduce spatial dimensions
        anchor_features = self.global_avg_pool(anchor_features).squeeze(-1).squeeze(-1)  # Shape: (batch_size, num_features)
        positive_features = self.global_avg_pool(positive_features).squeeze(-1).squeeze(-1)
        negative_features = self.global_avg_pool(negative_features).squeeze(-1).squeeze(-1)

        # Project features into the embedding space
        anchor_embedding = self.embedding_layer(anchor_features)
        positive_embedding = self.embedding_layer(positive_features)
        negative_embedding = self.embedding_layer(negative_features)

        return anchor_embedding, positive_embedding, negative_embedding

# Function to initialize the model
def get_embedding_model(num_classes=2):
    model = TripletSignatureEmbeddingModel(num_classes=num_classes)
    return model

# Custom loss function for triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        # Compute triplet loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


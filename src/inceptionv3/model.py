import torch
import torch.nn as nn
from timm import create_model

class TripletSignatureEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256, pretrained=True):
        super(TripletSignatureEmbeddingModel, self).__init__()
        # Load the pretrained Inception V3 model from timm
        self.inception = create_model('inception_v3', pretrained=pretrained)
        # Replace the classifier (fully connected layer) with an embedding layer
        in_features = self.inception.get_classifier().in_features
        self.inception.fc = nn.Identity()  # Remove the final classification layer
        self.embedding_layer = nn.Linear(in_features, embedding_size)  # Add new embedding layer

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.extract_features(anchor)
        positive_embedding = self.extract_features(positive)
        negative_embedding = self.extract_features(negative)
        return anchor_embedding, positive_embedding, negative_embedding

    def extract_features(self, x):
        # Pass the input through Inception to get feature embeddings
        base_features = self.inception(x)
        embedding = self.embedding_layer(base_features)
        return embedding

def get_embedding_model(embedding_size=256):
    model = TripletSignatureEmbeddingModel(embedding_size=embedding_size)
    return model

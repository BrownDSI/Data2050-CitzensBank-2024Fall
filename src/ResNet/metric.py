import numpy as np
import torch
from sklearn.metrics import roc_curve, confusion_matrix, fbeta_score

# Function to calculate EER
def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0) 
    frr = 1 - tpr
    try:
        eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - frr)))]
        eer = fpr[np.nanargmin(np.absolute((fpr - frr)))]
    except ValueError:
        eer_threshold = None
        eer = None
    return eer, eer_threshold


def evaluate(loader, model, beta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for anchors, positives, negatives in loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors, positives, negatives)
            positive_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings)
            negative_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings)
            labels = np.concatenate([np.ones(len(positive_distances)), np.zeros(len(negative_distances))])
            scores = np.concatenate([positive_distances.cpu().numpy(), negative_distances.cpu().numpy()])
            all_labels.extend(labels)
            all_scores.extend(scores)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=0)
    frr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - frr)))]
    eer = fpr[np.nanargmin(np.absolute((fpr - frr)))]

    predictions = all_scores <= eer_threshold
    fbeta = fbeta_score(all_labels, predictions, beta=beta)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    ks_statistic = np.max(tpr - fpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    return eer, eer_threshold, fbeta, ks_statistic, optimal_threshold, tp, fp, fn, tn
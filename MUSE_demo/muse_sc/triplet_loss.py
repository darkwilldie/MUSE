import torch
import torch.nn.functional as F


def pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.T)
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: torch.BoolTensor with shape [batch_size, batch_size]
    """
    device = labels.device
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = indices_not_equal & labels_equal

    return mask


def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: torch.BoolTensor with shape [batch_size, batch_size]
    """
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = ~labels_equal

    return mask


def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.Tensor with shape [batch_size]
    """
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = i_equal_j & ~i_equal_k

    mask = distinct_indices & valid_labels

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels).float()
    triplet_loss = mask * triplet_loss
    triplet_loss = torch.clamp(triplet_loss, min=0.0)

    valid_triplets = triplet_loss > 1e-16
    num_positive_triplets = valid_triplets.sum().float()
    num_valid_triplets = mask.sum().float()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    mask_anchor_positive = (
        get_anchor_positive_triplet_mask(labels).float()
    )
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist = anchor_positive_dist.max(1, keepdim=True)[0]

    mask_anchor_negative = (
        get_anchor_negative_triplet_mask(labels).float()
    )
    max_anchor_negative_dist = pairwise_dist.max(1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
        1.0 - mask_anchor_negative
    )
    hardest_negative_dist = anchor_negative_dist.min(1, keepdim=True)[0]

    triplet_loss = torch.clamp(
        hardest_positive_dist - hardest_negative_dist + margin, min=0.0
    )
    triplet_loss = triplet_loss.mean()

    return triplet_loss

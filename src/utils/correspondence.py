from __future__ import annotations

import torch


def compute_local_edge_features(
    points: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute relative position features for local (same-type) KNN edges.

    Args:
        points: [N, 3] point coordinates.
        knn_idx: [N, K] neighbor indices.
        knn_valid_mask: [N, K] bool mask for valid neighbors.

    Returns:
        rel_xyz: [N, K, 3] relative displacement from query to neighbor.
        rel_dist: [N, K, 1] Euclidean distance.
    """
    N, K = knn_idx.shape
    # Gather neighbor points: [N, K, 3]
    neighbor_points = gather_knn_features(points, knn_idx, knn_valid_mask)
    # Query points: [N, 1, 3]
    query_points = points.unsqueeze(1)
    # Relative displacement: [N, K, 3]
    rel_xyz = neighbor_points - query_points
    rel_dist = torch.norm(rel_xyz, dim=-1, keepdim=True)  # [N, K, 1]
    return rel_xyz, rel_dist


def compute_cross_edge_features(
    hand_points: torch.Tensor,
    obj_points: torch.Tensor,
    hand_normals: torch.Tensor,
    obj_normals: torch.Tensor,
    hand_to_obj_knn_idx: torch.Tensor,
    hand_to_obj_knn_valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute geometric features for hand-to-object cross edges.

    Args:
        hand_points: [Nh, 3] hand point coordinates.
        obj_points: [No, 3] object point coordinates.
        hand_normals: [Nh, 3] hand point normals.
        obj_normals: [No, 3] object point normals.
        hand_to_obj_knn_idx: [Nh, K_cross] object neighbor indices.
        hand_to_obj_knn_valid_mask: [Nh, K_cross] valid mask.

    Returns:
        dict with:
            delta: [Nh, K_cross, 3] hand_point - obj_point.
            dist: [Nh, K_cross, 1] Euclidean distance.
            signed_dist: [Nh, K_cross, 1] dot(obj_normal, delta).
            normal_dot: [Nh, K_cross, 1] dot(obj_normal, hand_normal).
    """
    Nh, K_cross = hand_to_obj_knn_idx.shape

    # Gather neighbor object points and normals: [Nh, K_cross, 3]
    neighbor_obj_points = gather_knn_features(obj_points, hand_to_obj_knn_idx, hand_to_obj_knn_valid_mask)
    neighbor_obj_normals = gather_knn_features(obj_normals, hand_to_obj_knn_idx, hand_to_obj_knn_valid_mask)

    # Hand query points: [Nh, 1, 3]
    query_hand_points = hand_points.unsqueeze(1)
    query_hand_normals = hand_normals.unsqueeze(1)

    # delta = hand_point - obj_point: [Nh, K_cross, 3]
    delta = query_hand_points - neighbor_obj_points

    # dist = ||delta||: [Nh, K_cross, 1]
    dist = torch.norm(delta, dim=-1, keepdim=True)

    # signed_dist = dot(obj_normal, delta): [Nh, K_cross, 1]
    signed_dist = torch.sum(neighbor_obj_normals * delta, dim=-1, keepdim=True)

    # normal_dot = dot(obj_normal, hand_normal): [Nh, K_cross, 1]
    normal_dot = torch.sum(neighbor_obj_normals * query_hand_normals, dim=-1, keepdim=True)

    return {
        "delta": delta,
        "dist": dist,
        "signed_dist": signed_dist,
        "normal_dot": normal_dot,
    }


def compute_obj_to_hand_edge_features(
    obj_points: torch.Tensor,
    hand_points: torch.Tensor,
    obj_normals: torch.Tensor,
    hand_normals: torch.Tensor,
    obj_to_hand_knn_idx: torch.Tensor,
    obj_to_hand_knn_valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute geometric features for object-to-hand cross edges.

    Args:
        obj_points: [No, 3] object point coordinates.
        hand_points: [Nh, 3] hand point coordinates.
        obj_normals: [No, 3] object point normals.
        hand_normals: [Nh, 3] hand point normals.
        obj_to_hand_knn_idx: [No, K_cross] hand neighbor indices.
        obj_to_hand_knn_valid_mask: [No, K_cross] valid mask.

    Returns:
        dict with:
            delta: [No, K_cross, 3] hand_point - obj_point.
            dist: [No, K_cross, 1] Euclidean distance.
            signed_dist: [No, K_cross, 1] dot(obj_normal, delta).
            normal_dot: [No, K_cross, 1] dot(obj_normal, hand_normal).
    """
    neighbor_hand_points = gather_knn_features(hand_points, obj_to_hand_knn_idx, obj_to_hand_knn_valid_mask)
    neighbor_hand_normals = gather_knn_features(hand_normals, obj_to_hand_knn_idx, obj_to_hand_knn_valid_mask)

    query_obj_points = obj_points.unsqueeze(1)
    query_obj_normals = obj_normals.unsqueeze(1)

    delta = neighbor_hand_points - query_obj_points
    dist = torch.norm(delta, dim=-1, keepdim=True)
    signed_dist = torch.sum(query_obj_normals * delta, dim=-1, keepdim=True)
    normal_dot = torch.sum(query_obj_normals * neighbor_hand_normals, dim=-1, keepdim=True)

    return {
        "delta": delta,
        "dist": dist,
        "signed_dist": signed_dist,
        "normal_dot": normal_dot,
    }


def soft_contact_label(
    dist: torch.Tensor,
    d_pos: float = 0.005,
    d_neg: float = 0.03,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute soft contact label from distances.

    Args:
        dist: [*] distances.
        d_pos: positive threshold (5mm).
        d_neg: negative threshold (3cm).
        gamma: decay exponent.

    Returns:
        label: [*] soft labels in [0, 1].
    """
    label = torch.zeros_like(dist)
    pos_mask = dist <= d_pos
    mid_mask = (dist > d_pos) & (dist < d_neg)
    label[pos_mask] = 1.0
    if mid_mask.any():
        alpha = 1.0 - (dist[mid_mask] - d_pos) / (d_neg - d_pos)
        label[mid_mask] = torch.pow(alpha.clamp(0.0, 1.0), gamma)
    return label


def contact_prob_to_bins(
    contact_prob: torch.Tensor,
    *,
    num_bins: int = 10,
) -> torch.Tensor:
    """Map continuous contact probabilities in [0, 1] to integer bins.

    This follows the ContactOpt convention:
      [0.0, 0.1) -> 0
      [0.1, 0.2) -> 1
      ...
      [0.9, 1.0] -> 9
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    scaled = torch.floor(contact_prob.clamp(min=0.0, max=1.0) * float(num_bins))
    return scaled.clamp(min=0, max=num_bins - 1).long()


def decode_contact_bin_logits(
    logits: torch.Tensor,
    *,
    mode: str = "expectation",
) -> torch.Tensor:
    """Decode per-bin logits back to a scalar contact probability in [0, 1]."""
    if logits.shape[-1] <= 0:
        raise ValueError("Last dimension of logits must be positive.")
    num_bins = int(logits.shape[-1])
    if mode == "expectation":
        prob = torch.softmax(logits, dim=-1)
        centers = (
            torch.arange(
                num_bins,
                device=logits.device,
                dtype=logits.dtype,
            )
            + 0.5
        ) / float(num_bins)
        return torch.sum(prob * centers, dim=-1)
    if mode == "argmax":
        cls = torch.argmax(logits, dim=-1)
        return (cls.to(dtype=logits.dtype) + 0.5) / float(num_bins)
    raise ValueError(f"Unsupported contact bin decode mode: {mode!r}")


def gather_knn_features(
    features: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather features from KNN neighbors.

    Args:
        features: [N, C] source features.
        knn_idx: [N, K] neighbor indices (may contain -1 for padding).
        knn_valid_mask: [N, K] bool mask for valid neighbors.

    Returns:
        gathered: [N, K, C] gathered features, zero for invalid neighbors.
    """
    knn_idx = knn_idx.long()
    N, K = knn_idx.shape
    C = features.shape[-1]

    # Clamp negative indices to 0 for safe gather, then mask
    safe_idx = knn_idx.clamp(min=0)  # [N, K]
    gathered = features[safe_idx]  # [N, K, C]

    # Zero out invalid neighbors
    mask = knn_valid_mask.unsqueeze(-1).float()  # [N, K, 1]
    gathered = gathered * mask

    return gathered

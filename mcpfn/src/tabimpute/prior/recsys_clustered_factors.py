"""Synthetic clustered preference factors; no observed-data inputs."""
import math

import torch


def clustered_vectors(count, rank, clusters, strength, generator, device="cpu"):
    if count < 1 or rank < 1 or clusters < 2 or not 0 <= strength <= 1:
        raise ValueError("Require positive factor dimensions, at least two clusters, and strength in [0,1]")
    centers = torch.randn(clusters, rank, generator=generator, device=device)
    centers = (centers - centers.mean(0)) / math.sqrt(1 - 1 / clusters)
    assignments = torch.randint(clusters, (count,), generator=generator, device=device)
    residual = torch.randn(count, rank, generator=generator, device=device)
    vectors = math.sqrt(strength) * centers[assignments] + math.sqrt(1 - strength) * residual
    return vectors, assignments


def clustered_affinity(rows, cols, rank, generator, device="cpu"):
    user_clusters = int(torch.randint(2, 17, (), generator=generator, device=device))
    item_clusters = int(torch.randint(2, 17, (), generator=generator, device=device))
    strength = .5 + .45 * torch.rand((), generator=generator, device=device).item()
    users, _ = clustered_vectors(rows, rank, user_clusters, strength, generator, device)
    items, _ = clustered_vectors(cols, rank, item_clusters, strength, generator, device)
    affinity = users @ items.T / math.sqrt(rank)
    return affinity, {"user_clusters": user_clusters, "item_clusters": item_clusters,
                      "cluster_strength": strength}

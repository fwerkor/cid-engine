from __future__ import annotations

import math

import torch
from torch import Tensor


def live_slot_occupancy(
    slot_occupancy: Tensor,
    lifecycle_features: Tensor | None,
    retired_index: int,
) -> Tensor:
    occupancy = slot_occupancy.clamp(0.0, 1.0)
    if lifecycle_features is None:
        return occupancy
    retired = lifecycle_features[..., retired_index : retired_index + 1].clamp(0.0, 1.0)
    return occupancy * (1.0 - retired)


def prefix_allocation_mask(
    occupancy: Tensor,
    allocation_logits: Tensor,
    threshold: float,
    max_allocations: int,
) -> Tensor:
    if occupancy.ndim == 3:
        occupancy = occupancy.squeeze(-1)
    probabilities = torch.sigmoid(allocation_logits.float())
    result = torch.zeros_like(occupancy, dtype=torch.bool)

    for batch in range(occupancy.shape[0]):
        allocations = 0
        for slot in range(occupancy.shape[1]):
            if bool(occupancy[batch, slot]):
                continue
            if float(probabilities[batch, slot]) < threshold:
                break
            if allocations < max_allocations:
                result[batch, slot] = True
                allocations += 1
    return result


def thought_corrupt_from_epsilon(
    semantic: Tensor,
    timesteps: Tensor,
    occupancy: Tensor,
    epsilon: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    alpha = torch.cos(timesteps * (math.pi / 2)).square()
    alpha = alpha.to(dtype=semantic.dtype).unsqueeze(-1)
    corrupted = alpha.sqrt() * semantic + (1.0 - alpha).sqrt() * epsilon
    occupied = occupancy.bool()
    corrupted = torch.where(occupied, corrupted, torch.zeros_like(corrupted))
    masked_epsilon = torch.where(occupied, epsilon, torch.zeros_like(epsilon))
    local_noise = timesteps.to(dtype=semantic.dtype).unsqueeze(-1)
    local_noise = torch.where(occupied, local_noise, torch.zeros_like(local_noise))
    return corrupted, local_noise, masked_epsilon


def display_token_statistics(
    token_ids: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    current_confidence = probabilities.gather(
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)
    return confidence, predicted, current_confidence

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
    occupied = occupancy.bool()
    eligible = torch.sigmoid(allocation_logits.float()) >= threshold
    free = ~occupied
    blocked = (free & ~eligible).cumsum(dim=1) > 0
    selected = free & eligible & ~blocked
    allocation_rank = selected.cumsum(dim=1)
    return selected & (allocation_rank <= max_allocations)


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
    zeros = torch.zeros((), dtype=semantic.dtype, device=semantic.device)
    corrupted = torch.where(occupied, corrupted, zeros)
    masked_epsilon = torch.where(occupied, epsilon, zeros)
    local_noise = timesteps.to(dtype=semantic.dtype).unsqueeze(-1)
    local_noise = torch.where(occupied, local_noise, zeros)
    return corrupted, local_noise, masked_epsilon


def display_token_statistics(
    token_ids: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    logits_f32 = logits.float()
    max_logits, predicted = logits_f32.max(dim=-1)
    normalizer = torch.logsumexp(logits_f32, dim=-1)
    confidence = torch.exp(max_logits - normalizer)
    current_logits = logits_f32.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    current_confidence = torch.exp(current_logits - normalizer)
    return confidence, predicted, current_confidence

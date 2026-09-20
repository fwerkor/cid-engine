from __future__ import annotations

import math
from bisect import bisect_right

import torch
from torch import Tensor

_MAX_STRUCTURAL_EDIT_TOKENS = 32


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
    occupied = occupancy.bool()
    eligible = probabilities >= threshold
    free = ~occupied
    blocked = (free & ~eligible).cumsum(dim=1) > 0
    selected = free & eligible & ~blocked
    allocation_rank = selected.cumsum(dim=1)
    return selected & (allocation_rank <= max_allocations)



def batched_linear_assignment(costs: Tensor, row_counts: Tensor) -> Tensor:
    if costs.ndim != 3:
        raise ValueError("costs must have shape [batch, rows, columns]")
    batch, rows, columns = costs.shape
    if rows > columns:
        raise ValueError("assignment rows cannot exceed columns")
    if row_counts.shape != (batch,):
        raise ValueError("row_counts must have shape [batch]")

    matrices = costs.detach().float().cpu().tolist()
    counts = row_counts.detach().long().cpu().tolist()
    result = torch.full(
        (batch, rows),
        -1,
        dtype=torch.long,
        device=costs.device,
    )
    assignments: list[list[int]] = []
    for matrix, active_rows in zip(matrices, counts, strict=True):
        if not 0 <= active_rows <= rows:
            raise ValueError("row_counts values must be in [0, rows]")
        u = [0.0] * (active_rows + 1)
        v = [0.0] * (columns + 1)
        p = [0] * (columns + 1)
        way = [0] * (columns + 1)
        for row in range(1, active_rows + 1):
            p[0] = row
            column0 = 0
            minimum = [float("inf")] * (columns + 1)
            used = [False] * (columns + 1)
            while True:
                used[column0] = True
                row0 = p[column0]
                delta = float("inf")
                column1 = 0
                for column in range(1, columns + 1):
                    if used[column]:
                        continue
                    current = matrix[row0 - 1][column - 1] - u[row0] - v[column]
                    if current < minimum[column]:
                        minimum[column] = current
                        way[column] = column0
                    if minimum[column] < delta:
                        delta = minimum[column]
                        column1 = column
                for column in range(columns + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        minimum[column] -= delta
                column0 = column1
                if p[column0] == 0:
                    break
            while True:
                column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1
                if column0 == 0:
                    break

        assignment = [-1] * rows
        for column in range(1, columns + 1):
            if 0 < p[column] <= active_rows:
                assignment[p[column] - 1] = column - 1
        assignments.append(assignment)

    if assignments:
        return torch.tensor(assignments, dtype=torch.long, device=costs.device)
    return result


def rollout_slot_transition(
    occupancy: Tensor,
    allocation_logits: Tensor,
    lifecycle_logits: Tensor,
    revision_logits: Tensor,
    input_lifecycle: Tensor,
    threshold: float,
    max_allocations: int,
    retired_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if occupancy.ndim == 3:
        occupancy = occupancy.squeeze(-1)
    occupied = occupancy.bool()
    allocation = prefix_allocation_mask(
        occupied,
        allocation_logits,
        threshold,
        max_allocations,
    )
    next_occupancy = occupied | allocation
    lifecycle_indices = lifecycle_logits.argmax(dim=-1)
    revision_indices = revision_logits.argmax(dim=-1)
    input_lifecycle_indices = input_lifecycle.argmax(dim=-1)
    input_lifecycle_present = input_lifecycle.abs().sum(dim=-1).ne(0)
    previous_retired = occupied & (input_lifecycle_indices == retired_index)
    newly_allocated = next_occupancy & ~occupied
    predicted_retired = lifecycle_indices == retired_index
    live_slots = next_occupancy & ~previous_retired & (~predicted_retired | newly_allocated)
    return (
        next_occupancy,
        lifecycle_indices,
        revision_indices,
        input_lifecycle_indices,
        input_lifecycle_present,
        live_slots,
    )


def masked_diffusion_corrupt_from_random(
    clean_ids: Tensor,
    ratio_random: Tensor,
    mask_random: Tensor,
    mask_token_id: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if clean_ids.ndim != 2:
        raise ValueError("clean_ids must have shape [batch, tokens]")
    batch, tokens = clean_ids.shape
    if ratio_random.shape != (batch, 1):
        raise ValueError("ratio_random must have shape [batch, 1]")
    if mask_random.shape != clean_ids.shape:
        raise ValueError("mask_random must match clean_ids shape")
    if not 0.0 < min_mask_ratio <= max_mask_ratio <= 1.0:
        raise ValueError("mask ratio range must satisfy 0 < min <= max <= 1")

    ratio_f32 = ratio_random.float()
    mask_random_f32 = mask_random.float()
    mask_ratio = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * ratio_f32
    masked = mask_random_f32 < mask_ratio
    empty_rows = ~masked.any(dim=1)
    fallback_positions = mask_random_f32.argmin(dim=1, keepdim=True)
    fallback = torch.zeros_like(masked)
    fallback.scatter_(1, fallback_positions, empty_rows.unsqueeze(1))
    masked |= fallback
    corrupted = clean_ids.masked_fill(masked, int(mask_token_id))
    metrics = torch.stack((masked.float().mean(), mask_ratio.mean()))
    return corrupted, masked, mask_ratio, metrics


def display_corrupt_from_random(
    token_ids: Tensor,
    timesteps: Tensor,
    eligible_mask: Tensor,
    corruption_random: Tensor,
    replacement_random: Tensor | None,
    replacement_offsets: Tensor | None,
    mask_token_id: int,
    eos_token_id: int | None,
    vocab_size: int,
    replacement_fraction: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, tokens]")
    batch, _ = token_ids.shape
    if timesteps.shape != (batch,):
        raise ValueError("timesteps must have shape [batch]")
    if eligible_mask.shape != token_ids.shape:
        raise ValueError("eligible_mask must match token_ids shape")
    if corruption_random.shape != token_ids.shape:
        raise ValueError("corruption_random must match token_ids shape")
    if not 0.0 <= replacement_fraction <= 1.0:
        raise ValueError("replacement_fraction must be in [0, 1]")

    eligible = eligible_mask.bool()
    corrupted_positions = (corruption_random.float() < timesteps.float()[:, None]) & eligible
    empty_rows = ~corrupted_positions.any(dim=1)
    fallback_rows = empty_rows & (timesteps.float() > 0.0) & eligible.any(dim=1)
    first_eligible = eligible.long().argmax(dim=1, keepdim=True)
    fallback = torch.zeros_like(corrupted_positions)
    fallback.scatter_(1, first_eligible, fallback_rows.unsqueeze(1))
    corrupted_positions |= fallback

    if replacement_fraction:
        if vocab_size < 3:
            raise ValueError("visible replacement corruption requires vocab_size >= 3")
        if replacement_random is None or replacement_offsets is None:
            raise ValueError(
                "replacement random tensors are required when replacement_fraction > 0"
            )
        replaced = corrupted_positions & (replacement_random.float() < replacement_fraction)
        replacements = (token_ids + replacement_offsets) % vocab_size
        for _ in range(3):
            forbidden = (replacements == mask_token_id) | (replacements == token_ids)
            if eos_token_id is not None:
                forbidden |= replacements == eos_token_id
            replacements = torch.where(forbidden, (replacements + 1) % vocab_size, replacements)
        masked = corrupted_positions & ~replaced
        corrupted = token_ids.masked_fill(masked, int(mask_token_id))
        corrupted = torch.where(replaced, replacements, corrupted)
    else:
        replaced = torch.zeros_like(corrupted_positions)
        masked = corrupted_positions
        corrupted = token_ids.masked_fill(masked, int(mask_token_id))
    labels = torch.where(
        corrupted_positions,
        token_ids,
        torch.full((), -100, dtype=token_ids.dtype, device=token_ids.device),
    )
    return corrupted, labels, masked, replaced


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


def materialize_cell_snapshot(
    thought_semantic: Tensor,
    role_logits: Tensor,
    uncertainty: Tensor,
    noise_delta: Tensor,
    lifecycle_logits: Tensor,
    selected: Tensor,
    semantic_indices: Tensor,
) -> Tensor:
    selected_f32 = selected.float().unsqueeze(-1)
    lifecycle = lifecycle_logits.float().argmax(dim=-1, keepdim=True).float()
    roles = torch.sigmoid(role_logits.float())
    semantic = thought_semantic.float().index_select(
        -1,
        semantic_indices.to(device=thought_semantic.device),
    )
    return torch.cat(
        (
            selected_f32,
            lifecycle,
            uncertainty.float(),
            noise_delta.float(),
            roles,
            semantic,
        ),
        dim=-1,
    )


def display_token_statistics(
    token_ids: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    logits_f32 = logits.float()
    probabilities = torch.softmax(logits_f32, dim=-1)
    predicted = logits_f32.argmax(dim=-1)
    confidence = probabilities.gather(
        dim=-1,
        index=predicted.unsqueeze(-1),
    ).squeeze(-1)
    current_confidence = probabilities.gather(
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)
    return confidence, predicted, current_confidence


def _structural_display_edit(
    token_ids: Tensor,
    predicted: Tensor,
    gains: Tensor,
    *,
    eos_position: int,
    revision_fraction: float,
    revision_margin: float,
    mask_token_id: int,
    eos_token_id: int,
) -> tuple[int, int, tuple[int, ...]] | None:
    if eos_position <= 0:
        return None

    active = token_ids[: eos_position + 1].tolist()
    proposal = predicted.detach().tolist()
    gain_values = gains[:eos_position].detach().tolist()
    edit_budget = min(
        _MAX_STRUCTURAL_EDIT_TOKENS,
        max(1, math.ceil(eos_position * revision_fraction)),
    )
    capacity = int(token_ids.shape[0])
    candidates: list[
        tuple[tuple[int, float, int, int], int, int, tuple[int, ...]]
    ] = []

    def matched_prefix(left: list[int], right: list[int]) -> int:
        count = 0
        for left_token, right_token in zip(left, right, strict=False):
            if left_token != right_token:
                break
            count += 1
        return count

    def anchor_positions(sequence: list[int]) -> dict[int, dict[tuple[int, ...], list[int]]]:
        result: dict[int, dict[tuple[int, ...], list[int]]] = {2: {}, 3: {}}
        for length in (2, 3):
            for position in range(len(sequence) - length + 1):
                key = tuple(sequence[position : position + length])
                result[length].setdefault(key, []).append(position)
        return result

    def nearest_after(positions: list[int], start: int, maximum: int) -> int | None:
        index = bisect_right(positions, start)
        if index >= len(positions) or positions[index] > maximum:
            return None
        return positions[index]

    def add_candidate(
        *,
        start: int,
        delete_count: int,
        inserted: tuple[int, ...],
        anchor: int,
    ) -> None:
        if anchor < 2 or gain_values[start] < revision_margin:
            return
        net_growth = len(inserted) - delete_count
        if eos_position + 1 + net_growth > capacity:
            return
        score = (anchor, float(gain_values[start]), -abs(net_growth), start)
        candidates.append((score, start, delete_count, inserted))

    changed_starts = [
        start
        for start in range(eos_position)
        if proposal[start] != active[start] and gain_values[start] >= revision_margin
    ]
    if not changed_starts:
        return None

    active_anchors = anchor_positions(active)
    proposal_anchors = anchor_positions(proposal)
    for start in changed_starts:
        suffix_length = len(active) - start
        anchor_length = min(3, suffix_length)
        insertion_key = tuple(active[start : start + anchor_length])
        insertion_anchor_start = nearest_after(
            proposal_anchors[anchor_length].get(insertion_key, []),
            start,
            start + edit_budget,
        )
        if insertion_anchor_start is not None:
            inserted = tuple(proposal[start:insertion_anchor_start])
            if inserted and not any(
                token == mask_token_id or token == eos_token_id for token in inserted
            ):
                add_candidate(
                    start=start,
                    delete_count=0,
                    inserted=inserted,
                    anchor=matched_prefix(
                        proposal[insertion_anchor_start:],
                        active[start:],
                    ),
                )

        anchor_length = min(3, len(proposal) - start)
        if anchor_length < 2:
            continue
        deletion_key = tuple(proposal[start : start + anchor_length])
        deletion_suffix_start = nearest_after(
            active_anchors[anchor_length].get(deletion_key, []),
            start,
            min(eos_position - 1, start + edit_budget),
        )
        if deletion_suffix_start is not None:
            add_candidate(
                start=start,
                delete_count=deletion_suffix_start - start,
                inserted=(),
                anchor=matched_prefix(
                    proposal[start:],
                    active[deletion_suffix_start:],
                ),
            )

    if not candidates:
        return None
    _, start, delete_count, inserted = max(candidates, key=lambda item: item[0])
    return start, delete_count, inserted


def refine_display_from_statistics(
    token_ids: Tensor,
    confidence: Tensor,
    predicted: Tensor,
    current_confidence: Tensor,
    *,
    mask_token_id: int,
    eos_token_id: int | None,
    reveal_fraction: float,
    revision_fraction: float,
    revision_margin: float,
) -> Tensor:
    """Pure-Python CID display refinement oracle used by differential tests."""
    result = token_ids.clone()
    for batch_index in range(token_ids.shape[0]):
        eos_position: int | None = None
        if eos_token_id is not None:
            current_eos = torch.nonzero(
                token_ids[batch_index] == eos_token_id,
                as_tuple=False,
            ).flatten()
            if current_eos.numel():
                eos_position = int(current_eos[0])

        structural_edit = None
        if eos_position is not None and revision_fraction:
            structural_edit = _structural_display_edit(
                token_ids[batch_index],
                predicted[batch_index],
                confidence[batch_index] - current_confidence[batch_index],
                eos_position=eos_position,
                revision_fraction=revision_fraction,
                revision_margin=revision_margin,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
            )
        if structural_edit is not None:
            start, delete_count, inserted = structural_edit
            active = token_ids[batch_index, : eos_position + 1].tolist()
            revised = [*active[:start], *inserted, *active[start + delete_count :]]
            result[batch_index].fill_(mask_token_id)
            result[batch_index, : len(revised)] = torch.tensor(
                revised,
                dtype=result.dtype,
                device=result.device,
            )
            continue

        active_content_stop = eos_position if eos_position is not None else token_ids.shape[1]
        masked_positions = torch.nonzero(
            token_ids[batch_index, :active_content_stop] == mask_token_id,
            as_tuple=False,
        ).flatten()
        if masked_positions.numel() and reveal_fraction:
            reveal_count = math.ceil(masked_positions.numel() * reveal_fraction)
            ranked = masked_positions[
                confidence[batch_index, masked_positions].argsort(
                    descending=True,
                    stable=True,
                )
            ]
            selected = ranked[:reveal_count]
            result[batch_index, selected] = predicted[batch_index, selected]

        if revision_fraction == 0.0:
            continue
        visible_positions = torch.nonzero(
            token_ids[batch_index, :active_content_stop] != mask_token_id,
            as_tuple=False,
        ).flatten()
        if visible_positions.numel():
            current_ids = token_ids[batch_index, visible_positions]
            visible_confidence = current_confidence[batch_index, visible_positions]
            gains = confidence[batch_index, visible_positions] - visible_confidence
            candidates = (predicted[batch_index, visible_positions] != current_ids) & (
                gains >= revision_margin
            )
            candidate_positions = visible_positions[candidates]
            if candidate_positions.numel():
                candidate_gains = gains[candidates]
                revision_count = min(
                    candidate_positions.numel(),
                    math.ceil(visible_positions.numel() * revision_fraction),
                )
                ranked = candidate_positions[
                    candidate_gains.argsort(descending=True, stable=True)
                ]
                selected = ranked[:revision_count]
                result[batch_index, selected] = predicted[batch_index, selected]

        if eos_position is not None and eos_position + 1 < token_ids.shape[1]:
            eos_prediction = int(predicted[batch_index, eos_position])
            eos_gain = float(
                confidence[batch_index, eos_position]
                - current_confidence[batch_index, eos_position]
            )
            if eos_prediction != eos_token_id and eos_gain >= revision_margin:
                expansion_budget = max(
                    1,
                    math.ceil(max(1, eos_position + 1) * reveal_fraction),
                )
                new_eos = min(
                    token_ids.shape[1] - 1,
                    eos_position + expansion_budget,
                )
                predicted_tail_eos = torch.nonzero(
                    predicted[batch_index, eos_position + 1 : new_eos + 1] == eos_token_id,
                    as_tuple=False,
                ).flatten()
                if predicted_tail_eos.numel():
                    new_eos = eos_position + 1 + int(predicted_tail_eos[0])
                result[batch_index, eos_position:new_eos] = predicted[
                    batch_index, eos_position:new_eos
                ]
                result[batch_index, new_eos] = eos_token_id

    if eos_token_id is not None:
        for batch_index in range(result.shape[0]):
            eos_positions = torch.nonzero(
                result[batch_index] == eos_token_id,
                as_tuple=False,
            ).flatten()
            if eos_positions.numel():
                result[batch_index, int(eos_positions[0]) + 1 :] = mask_token_id
    return result

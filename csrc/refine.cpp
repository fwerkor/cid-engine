#include "cid_engine/ops.hpp"

#include <ATen/Functions.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace cid::engine {

namespace {

struct StructuralEdit {
  std::int64_t start;
  std::int64_t delete_count;
  std::vector<std::int64_t> inserted;
};

struct StructuralCandidate {
  std::int64_t anchor;
  float gain;
  std::int64_t negative_abs_growth;
  std::int64_t start;
  std::int64_t delete_count;
  std::vector<std::int64_t> inserted;
};

bool candidate_better(
    const StructuralCandidate& left,
    const StructuralCandidate& right) {
  if (left.anchor != right.anchor) {
    return left.anchor > right.anchor;
  }
  if (left.gain != right.gain) {
    return left.gain > right.gain;
  }
  if (left.negative_abs_growth != right.negative_abs_growth) {
    return left.negative_abs_growth > right.negative_abs_growth;
  }
  return left.start > right.start;
}

std::int64_t matched_prefix(
    const std::vector<std::int64_t>& left,
    const std::int64_t left_start,
    const std::vector<std::int64_t>& right,
    const std::int64_t right_start) {
  std::int64_t count = 0;
  while (left_start + count < static_cast<std::int64_t>(left.size()) &&
         right_start + count < static_cast<std::int64_t>(right.size()) &&
         left[left_start + count] == right[right_start + count]) {
    ++count;
  }
  return count;
}

bool sequence_matches(
    const std::vector<std::int64_t>& sequence,
    const std::int64_t position,
    const std::vector<std::int64_t>& key) {
  if (position < 0 ||
      position + static_cast<std::int64_t>(key.size()) >
          static_cast<std::int64_t>(sequence.size())) {
    return false;
  }
  for (std::int64_t index = 0; index < static_cast<std::int64_t>(key.size()); ++index) {
    if (sequence[position + index] != key[index]) {
      return false;
    }
  }
  return true;
}

std::optional<std::int64_t> nearest_matching_position(
    const std::vector<std::int64_t>& sequence,
    const std::vector<std::int64_t>& key,
    const std::int64_t start,
    const std::int64_t maximum) {
  const auto last = std::min(
      maximum,
      static_cast<std::int64_t>(sequence.size()) -
          static_cast<std::int64_t>(key.size()));
  for (std::int64_t position = start + 1; position <= last; ++position) {
    if (sequence_matches(sequence, position, key)) {
      return position;
    }
  }
  return std::nullopt;
}

std::optional<StructuralEdit> structural_display_edit(
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::int64_t>& predicted,
    const std::vector<float>& gains,
    const std::int64_t eos_position,
    const double revision_fraction,
    const double revision_margin,
    const std::int64_t mask_token_id,
    const std::int64_t eos_token_id) {
  if (eos_position <= 0) {
    return std::nullopt;
  }

  std::vector<std::int64_t> active(
      token_ids.begin(),
      token_ids.begin() + eos_position + 1);
  const auto edit_budget = std::min<std::int64_t>(
      32,
      std::max<std::int64_t>(
          1,
          static_cast<std::int64_t>(
              std::ceil(static_cast<double>(eos_position) * revision_fraction))));
  const auto capacity = static_cast<std::int64_t>(token_ids.size());
  std::vector<StructuralCandidate> candidates;

  for (std::int64_t start = 0; start < eos_position; ++start) {
    if (predicted[start] == active[start] || gains[start] < revision_margin) {
      continue;
    }

    const auto insertion_anchor_length =
        std::min<std::int64_t>(3, static_cast<std::int64_t>(active.size()) - start);
    std::vector<std::int64_t> insertion_key(
        active.begin() + start,
        active.begin() + start + insertion_anchor_length);
    const auto insertion_anchor_start = nearest_matching_position(
        predicted,
        insertion_key,
        start,
        start + edit_budget);
    if (insertion_anchor_start.has_value()) {
      std::vector<std::int64_t> inserted(
          predicted.begin() + start,
          predicted.begin() + *insertion_anchor_start);
      const bool valid_insert =
          !inserted.empty() &&
          std::none_of(
              inserted.begin(),
              inserted.end(),
              [&](const std::int64_t token) {
                return token == mask_token_id || token == eos_token_id;
              });
      if (valid_insert) {
        const auto anchor = matched_prefix(
            predicted,
            *insertion_anchor_start,
            active,
            start);
        const auto net_growth = static_cast<std::int64_t>(inserted.size());
        if (anchor >= 2 &&
            gains[start] >= revision_margin &&
            eos_position + 1 + net_growth <= capacity) {
          candidates.push_back({
              anchor,
              gains[start],
              -std::abs(net_growth),
              start,
              0,
              std::move(inserted),
          });
        }
      }
    }

    const auto deletion_anchor_length = std::min<std::int64_t>(
        3,
        static_cast<std::int64_t>(predicted.size()) - start);
    if (deletion_anchor_length < 2) {
      continue;
    }
    std::vector<std::int64_t> deletion_key(
        predicted.begin() + start,
        predicted.begin() + start + deletion_anchor_length);
    const auto deletion_suffix_start = nearest_matching_position(
        active,
        deletion_key,
        start,
        std::min<std::int64_t>(eos_position - 1, start + edit_budget));
    if (deletion_suffix_start.has_value()) {
      const auto delete_count = *deletion_suffix_start - start;
      const auto anchor = matched_prefix(
          predicted,
          start,
          active,
          *deletion_suffix_start);
      const auto net_growth = -delete_count;
      if (anchor >= 2 &&
          gains[start] >= revision_margin &&
          eos_position + 1 + net_growth <= capacity) {
        candidates.push_back({
            anchor,
            gains[start],
            -std::abs(net_growth),
            start,
            delete_count,
            {},
        });
      }
    }
  }

  if (candidates.empty()) {
    return std::nullopt;
  }
  auto best = candidates.front();
  for (std::size_t index = 1; index < candidates.size(); ++index) {
    if (candidate_better(candidates[index], best)) {
      best = candidates[index];
    }
  }
  return StructuralEdit{
      best.start,
      best.delete_count,
      std::move(best.inserted),
  };
}

}  // namespace

at::Tensor refine_display_from_statistics(
    const at::Tensor& token_ids,
    const at::Tensor& confidence,
    const at::Tensor& predicted,
    const at::Tensor& current_confidence,
    const std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    const double reveal_fraction,
    const double revision_fraction,
    const double revision_margin) {
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(predicted.sizes() == token_ids.sizes(), "predicted must match token_ids");
  TORCH_CHECK(predicted.scalar_type() == at::kLong, "predicted must use torch.int64");
  TORCH_CHECK(confidence.sizes() == token_ids.sizes(), "confidence must match token_ids");
  TORCH_CHECK(
      current_confidence.sizes() == token_ids.sizes(),
      "current_confidence must match token_ids");
  TORCH_CHECK(reveal_fraction >= 0.0 && reveal_fraction <= 1.0, "reveal_fraction must be in [0, 1]");
  TORCH_CHECK(
      revision_fraction >= 0.0 && revision_fraction <= 1.0,
      "revision_fraction must be in [0, 1]");
  TORCH_CHECK(revision_margin >= 0.0, "revision_margin must be non-negative");

  const auto device = token_ids.device();
  auto tokens_cpu = token_ids.to(at::kCPU).contiguous();
  auto predicted_cpu = predicted.to(at::kCPU).contiguous();
  auto confidence_cpu = confidence.to(at::kCPU, at::kFloat).contiguous();
  auto current_cpu = current_confidence.to(at::kCPU, at::kFloat).contiguous();
  auto result_cpu = tokens_cpu.clone();

  const auto batch_size = token_ids.size(0);
  const auto token_count = token_ids.size(1);
  const auto* tokens = tokens_cpu.const_data_ptr<std::int64_t>();
  const auto* predictions = predicted_cpu.const_data_ptr<std::int64_t>();
  const auto* confidences = confidence_cpu.const_data_ptr<float>();
  const auto* currents = current_cpu.const_data_ptr<float>();
  auto* result = result_cpu.mutable_data_ptr<std::int64_t>();

  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const auto offset = batch * token_count;
    std::vector<std::int64_t> row_tokens(
        tokens + offset,
        tokens + offset + token_count);
    std::vector<std::int64_t> row_predicted(
        predictions + offset,
        predictions + offset + token_count);
    std::vector<float> row_confidence(
        confidences + offset,
        confidences + offset + token_count);
    std::vector<float> row_current(
        currents + offset,
        currents + offset + token_count);
    std::vector<float> gains(token_count);
    for (std::int64_t index = 0; index < token_count; ++index) {
      gains[index] = row_confidence[index] - row_current[index];
    }

    std::optional<std::int64_t> eos_position;
    if (eos_token_id.has_value()) {
      for (std::int64_t index = 0; index < token_count; ++index) {
        if (row_tokens[index] == *eos_token_id) {
          eos_position = index;
          break;
        }
      }
    }

    if (eos_position.has_value() && revision_fraction != 0.0) {
      auto edit = structural_display_edit(
          row_tokens,
          row_predicted,
          gains,
          *eos_position,
          revision_fraction,
          revision_margin,
          mask_token_id,
          *eos_token_id);
      if (edit.has_value()) {
        std::vector<std::int64_t> revised;
        revised.reserve(token_count);
        revised.insert(
            revised.end(),
            row_tokens.begin(),
            row_tokens.begin() + edit->start);
        revised.insert(
            revised.end(),
            edit->inserted.begin(),
            edit->inserted.end());
        revised.insert(
            revised.end(),
            row_tokens.begin() + edit->start + edit->delete_count,
            row_tokens.begin() + *eos_position + 1);
        std::fill(result + offset, result + offset + token_count, mask_token_id);
        std::copy(revised.begin(), revised.end(), result + offset);
        continue;
      }
    }

    const auto active_content_stop = eos_position.value_or(token_count);
    std::vector<std::int64_t> masked_positions;
    masked_positions.reserve(active_content_stop);
    for (std::int64_t index = 0; index < active_content_stop; ++index) {
      if (row_tokens[index] == mask_token_id) {
        masked_positions.push_back(index);
      }
    }
    if (!masked_positions.empty() && reveal_fraction != 0.0) {
      const auto reveal_count = static_cast<std::int64_t>(
          std::ceil(static_cast<double>(masked_positions.size()) * reveal_fraction));
      std::stable_sort(
          masked_positions.begin(),
          masked_positions.end(),
          [&](const std::int64_t left, const std::int64_t right) {
            return row_confidence[left] > row_confidence[right];
          });
      for (std::int64_t rank = 0; rank < reveal_count; ++rank) {
        const auto position = masked_positions[rank];
        result[offset + position] = row_predicted[position];
      }
    }

    if (revision_fraction != 0.0) {
      std::vector<std::int64_t> visible_positions;
      visible_positions.reserve(active_content_stop);
      for (std::int64_t index = 0; index < active_content_stop; ++index) {
        if (row_tokens[index] != mask_token_id) {
          visible_positions.push_back(index);
        }
      }
      std::vector<std::int64_t> candidates;
      for (const auto position : visible_positions) {
        if (row_predicted[position] != row_tokens[position] &&
            gains[position] >= revision_margin) {
          candidates.push_back(position);
        }
      }
      if (!candidates.empty()) {
        const auto revision_count = std::min<std::int64_t>(
            static_cast<std::int64_t>(candidates.size()),
            static_cast<std::int64_t>(
                std::ceil(static_cast<double>(visible_positions.size()) * revision_fraction)));
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [&](const std::int64_t left, const std::int64_t right) {
              return gains[left] > gains[right];
            });
        for (std::int64_t rank = 0; rank < revision_count; ++rank) {
          const auto position = candidates[rank];
          result[offset + position] = row_predicted[position];
        }
      }

      if (eos_position.has_value() && *eos_position + 1 < token_count) {
        const auto position = *eos_position;
        const auto eos_gain = gains[position];
        if (row_predicted[position] != *eos_token_id &&
            eos_gain >= revision_margin) {
          const auto expansion_budget = std::max<std::int64_t>(
              1,
              static_cast<std::int64_t>(
                  std::ceil(
                      static_cast<double>(std::max<std::int64_t>(1, position + 1)) *
                      reveal_fraction)));
          auto new_eos = std::min<std::int64_t>(
              token_count - 1,
              position + expansion_budget);
          for (std::int64_t tail = position + 1; tail <= new_eos; ++tail) {
            if (row_predicted[tail] == *eos_token_id) {
              new_eos = tail;
              break;
            }
          }
          for (std::int64_t tail = position; tail < new_eos; ++tail) {
            result[offset + tail] = row_predicted[tail];
          }
          result[offset + new_eos] = *eos_token_id;
        }
      }
    }
  }

  if (eos_token_id.has_value()) {
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const auto offset = batch * token_count;
      for (std::int64_t index = 0; index < token_count; ++index) {
        if (result[offset + index] == *eos_token_id) {
          std::fill(
              result + offset + index + 1,
              result + offset + token_count,
              mask_token_id);
          break;
        }
      }
    }
  }

  if (device.is_cpu()) {
    return result_cpu;
  }
  return result_cpu.to(device);
}

}  // namespace cid::engine

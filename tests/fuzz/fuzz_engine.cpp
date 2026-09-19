#include "cid_engine/ops.hpp"

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <tuple>
#include <vector>

namespace {

class Bytes {
 public:
  Bytes(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  std::uint8_t take() {
    if (offset_ >= size_) {
      return 0;
    }
    return data_[offset_++];
  }

  std::int64_t dim(std::int64_t maximum) {
    return 1 + static_cast<std::int64_t>(take() % maximum);
  }

  float finite_float(float scale = 8.0F) {
    const auto centered = static_cast<int>(take()) - 128;
    return scale * static_cast<float>(centered) / 128.0F;
  }

  double unit() {
    return static_cast<double>(take()) / 255.0;
  }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t offset_ = 0;
};

[[noreturn]] void fail() {
  std::abort();
}

void require(bool condition) {
  if (!condition) {
    fail();
  }
}

at::Tensor float_tensor(Bytes& bytes, const std::vector<std::int64_t>& shape) {
  std::int64_t count = 1;
  for (const auto extent : shape) {
    count *= extent;
  }
  std::vector<float> values(static_cast<std::size_t>(count));
  for (auto& value : values) {
    value = bytes.finite_float();
  }
  return at::tensor(values, at::TensorOptions().dtype(at::kFloat)).reshape(shape);
}

void fuzz_live(Bytes& bytes) {
  const auto batch = bytes.dim(3);
  const auto slots = bytes.dim(16);
  const auto features = bytes.dim(6);
  auto occupancy = float_tensor(bytes, {batch, slots, 1});
  const bool with_lifecycle = (bytes.take() & 1U) != 0;
  std::optional<at::Tensor> lifecycle = std::nullopt;
  const auto retired_index = static_cast<std::int64_t>(bytes.take() % features);
  if (with_lifecycle) {
    lifecycle = float_tensor(bytes, {batch, slots, features});
  }

  auto result = cid::engine::live_slot_occupancy(occupancy, lifecycle, retired_index);
  require(result.sizes() == occupancy.sizes());
  require(result.scalar_type() == occupancy.scalar_type());
  require(result.ge(0).logical_and(result.le(1)).all().item<bool>());
}

void fuzz_prefix(Bytes& bytes) {
  const auto batch = bytes.dim(3);
  const auto slots = bytes.dim(24);
  std::vector<float> occupancy_values(static_cast<std::size_t>(batch * slots));
  for (auto& value : occupancy_values) {
    value = static_cast<float>(static_cast<int>(bytes.take() % 3) - 1);
  }
  auto occupancy = at::tensor(occupancy_values, at::TensorOptions().dtype(at::kFloat))
                       .reshape({batch, slots});
  auto logits = float_tensor(bytes, {batch, slots});
  const auto threshold = bytes.unit();
  const auto max_allocations =
      1 + static_cast<std::int64_t>(bytes.take() % static_cast<std::uint8_t>(slots + 3));

  auto result = cid::engine::prefix_allocation_mask(
      occupancy,
      logits,
      threshold,
      max_allocations);
  require(result.sizes() == occupancy.sizes());
  require(result.scalar_type() == at::kBool);

  auto probabilities = logits.to(at::kFloat).sigmoid();
  auto expected = at::zeros_like(occupancy, at::TensorOptions().dtype(at::kBool));
  auto* expected_data = expected.mutable_data_ptr<bool>();
  const auto* occupancy_data = occupancy.const_data_ptr<float>();
  const auto* probability_data = probabilities.contiguous().const_data_ptr<float>();
  for (std::int64_t row = 0; row < batch; ++row) {
    std::int64_t allocations = 0;
    for (std::int64_t slot = 0; slot < slots; ++slot) {
      const auto index = row * slots + slot;
      if (occupancy_data[index] != 0.0F) {
        continue;
      }
      if (probability_data[index] < static_cast<float>(threshold)) {
        break;
      }
      if (allocations < max_allocations) {
        expected_data[index] = true;
        ++allocations;
      }
    }
  }
  require(result.equal(expected));
}

void fuzz_thought(Bytes& bytes) {
  const auto batch = bytes.dim(3);
  const auto slots = bytes.dim(12);
  const auto hidden = bytes.dim(32);
  auto semantic = float_tensor(bytes, {batch, slots, hidden});
  auto epsilon = float_tensor(bytes, {batch, slots, hidden});

  std::vector<float> timestep_values(static_cast<std::size_t>(batch * slots));
  for (auto& value : timestep_values) {
    value = static_cast<float>(bytes.unit());
  }
  auto timesteps = at::tensor(timestep_values, at::TensorOptions().dtype(at::kFloat))
                       .reshape({batch, slots});

  std::vector<float> occupancy_values(static_cast<std::size_t>(batch * slots));
  for (auto& value : occupancy_values) {
    value = (bytes.take() & 1U) != 0 ? 1.0F : 0.0F;
  }
  auto occupancy = at::tensor(occupancy_values, at::TensorOptions().dtype(at::kFloat))
                       .reshape({batch, slots, 1});

  auto [corrupted, local_noise, masked_epsilon] =
      cid::engine::thought_corrupt_from_epsilon(
          semantic,
          timesteps,
          occupancy,
          epsilon);
  require(corrupted.sizes() == semantic.sizes());
  require(masked_epsilon.sizes() == semantic.sizes());
  require(local_noise.sizes() == occupancy.sizes());
  require(at::isfinite(corrupted).all().item<bool>());
  require(at::isfinite(local_noise).all().item<bool>());
  require(at::isfinite(masked_epsilon).all().item<bool>());
}

void fuzz_display(Bytes& bytes) {
  const auto batch = bytes.dim(3);
  const auto tokens = bytes.dim(16);
  const auto vocab = bytes.dim(96);
  auto logits = float_tensor(bytes, {batch, tokens, vocab});

  std::vector<std::int64_t> ids(static_cast<std::size_t>(batch * tokens));
  for (auto& id : ids) {
    id = static_cast<std::int64_t>(bytes.take() % vocab);
  }
  auto token_ids = at::tensor(ids, at::TensorOptions().dtype(at::kLong))
                       .reshape({batch, tokens});

  auto [confidence, predicted, current] =
      cid::engine::display_token_statistics(token_ids, logits);
  auto probabilities = logits.to(at::kFloat).softmax(-1);
  auto reference_max = probabilities.max(-1);
  auto expected_confidence = std::get<0>(reference_max);
  auto expected_predicted = std::get<1>(reference_max);
  auto expected_current =
      probabilities.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1);

  require(predicted.equal(expected_predicted));
  require(at::allclose(confidence, expected_confidence, 3.0e-5, 2.0e-6));
  require(at::allclose(current, expected_current, 3.0e-5, 2.0e-6));
}

void fuzz_refine(Bytes& bytes) {
  const auto batch = bytes.dim(2);
  const auto tokens = bytes.dim(24);
  constexpr std::int64_t mask_token_id = 0;
  constexpr std::int64_t eos_token_id = 1;

  std::vector<std::int64_t> ids(static_cast<std::size_t>(batch * tokens));
  std::vector<std::int64_t> predictions(ids.size());
  std::vector<float> confidence_values(ids.size());
  std::vector<float> current_values(ids.size());

  for (std::size_t index = 0; index < ids.size(); ++index) {
    ids[index] = static_cast<std::int64_t>(bytes.take() % 32);
    predictions[index] = static_cast<std::int64_t>(bytes.take() % 32);
    confidence_values[index] = static_cast<float>(bytes.unit());
    current_values[index] =
        confidence_values[index] * static_cast<float>(bytes.unit());
  }

  auto token_ids = at::tensor(ids, at::TensorOptions().dtype(at::kLong))
                       .reshape({batch, tokens});
  auto predicted = at::tensor(predictions, at::TensorOptions().dtype(at::kLong))
                       .reshape({batch, tokens});
  auto confidence = at::tensor(confidence_values, at::TensorOptions().dtype(at::kFloat))
                        .reshape({batch, tokens});
  auto current = at::tensor(current_values, at::TensorOptions().dtype(at::kFloat))
                     .reshape({batch, tokens});

  const bool with_eos = (bytes.take() & 1U) != 0;
  const std::optional<std::int64_t> eos =
      with_eos ? std::optional<std::int64_t>(eos_token_id) : std::nullopt;
  if (with_eos) {
    for (std::int64_t row = 0; row < batch; ++row) {
      const auto eos_position =
          static_cast<std::int64_t>(bytes.take() % static_cast<std::uint8_t>(tokens));
      token_ids.index_put_({row, eos_position}, eos_token_id);
      if (eos_position + 1 < tokens) {
        token_ids.index_put_(
            {row, at::indexing::Slice(eos_position + 1, tokens)},
            mask_token_id);
      }
    }
  }

  auto result = cid::engine::refine_display_from_statistics(
      token_ids,
      confidence,
      predicted,
      current,
      mask_token_id,
      eos,
      bytes.unit(),
      bytes.unit(),
      bytes.unit());

  require(result.sizes() == token_ids.sizes());
  require(result.scalar_type() == at::kLong);
  if (with_eos) {
    auto result_cpu = result.contiguous();
    const auto* values = result_cpu.const_data_ptr<std::int64_t>();
    for (std::int64_t row = 0; row < batch; ++row) {
      bool seen_eos = false;
      for (std::int64_t column = 0; column < tokens; ++column) {
        const auto value = values[row * tokens + column];
        if (seen_eos) {
          require(value == mask_token_id);
        }
        if (value == eos_token_id) {
          seen_eos = true;
        }
      }
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  // Tiny tensor cases dominate this target.  Intra-op thread-pool startup costs
  // otherwise dwarf useful fuzzing work and sharply reduce executions/second.
  at::set_num_threads(1);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(
    const std::uint8_t* data,
    std::size_t size) {
  if (size == 0) {
    return 0;
  }
  Bytes bytes(data, size);
  try {
    switch (bytes.take() % 5U) {
      case 0:
        fuzz_live(bytes);
        break;
      case 1:
        fuzz_prefix(bytes);
        break;
      case 2:
        fuzz_thought(bytes);
        break;
      case 3:
        fuzz_display(bytes);
        break;
      default:
        fuzz_refine(bytes);
        break;
    }
  } catch (...) {
    fail();
  }
  return 0;
}

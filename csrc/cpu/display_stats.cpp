#include "cid_engine/ops.hpp"

#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace cid::engine {
namespace {

template <typename scalar_t>
void display_token_statistics_cpu_kernel(
    const scalar_t* logits,
    const std::int64_t* token_ids,
    const float* row_maxima,
    float* confidence,
    float* current_confidence,
    const std::int64_t rows,
    const std::int64_t vocab_size) {
  // A row is intentionally the unit of parallelism. CID commonly has tens to
  // hundreds of active token rows, while each row spans a large vocabulary.
  // Keeping one worker on a row avoids materializing the full probability
  // tensor and gives each worker sequential memory access.
  at::parallel_for(0, rows, 1, [&](const std::int64_t begin, const std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const scalar_t* row_logits = logits + row * vocab_size;
      const float row_max = row_maxima[row];
      const std::int64_t current_id = token_ids[row];
      TORCH_CHECK_INDEX(
          current_id >= 0 && current_id < vocab_size,
          "token id ", current_id, " is outside vocabulary size ", vocab_size);

      using fVec = at::vec::Vectorized<float>;
      const fVec row_max_vec(row_max);
      const float denominator = at::vec::map_reduce_all<scalar_t>(
          [row_max_vec](fVec values) { return (values - row_max_vec).exp(); },
          [](fVec left, fVec right) { return left + right; },
          row_logits,
          vocab_size);

      const float current_numerator = std::exp(
          static_cast<float>(row_logits[current_id]) - row_max);
      confidence[row] = 1.0F / denominator;
      current_confidence[row] = current_numerator / denominator;
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> display_token_statistics_cpu(
    const at::Tensor& token_ids,
    const at::Tensor& logits) {
  TORCH_CHECK(token_ids.device().is_cpu(), "token_ids must be CPU tensors");
  TORCH_CHECK(logits.device().is_cpu(), "logits must be CPU tensors");
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(
      logits.dim() == 3 &&
          logits.size(0) == token_ids.size(0) &&
          logits.size(1) == token_ids.size(1),
      "logits must have shape [batch, tokens, vocab]");
  TORCH_CHECK(logits.is_floating_point(), "logits must use a floating dtype");
  TORCH_CHECK(logits.size(-1) > 0, "logits vocabulary dimension must be non-empty");

  const auto contiguous_logits = logits.contiguous();
  const auto contiguous_ids = token_ids.contiguous();
  const std::int64_t rows = token_ids.numel();
  const std::int64_t vocab_size = logits.size(-1);

  auto confidence = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));
  auto current_confidence =
      at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));

  if (rows == 0) {
    auto predicted = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kLong));
    return {confidence, predicted, current_confidence};
  }

  // ATen's max reduction already has architecture-specific vectorized argmax
  // kernels. Reuse it for the first pass, then fuse only the probability
  // statistics that would otherwise require materializing a full softmax.
  auto max_result = contiguous_logits.max(-1);
  auto row_maxima = std::get<0>(max_result).to(at::kFloat).contiguous();
  auto predicted = std::get<1>(max_result);

  switch (contiguous_logits.scalar_type()) {
    case at::kFloat:
      display_token_statistics_cpu_kernel<float>(
          contiguous_logits.const_data_ptr<float>(),
          contiguous_ids.const_data_ptr<std::int64_t>(),
          row_maxima.const_data_ptr<float>(),
          confidence.mutable_data_ptr<float>(),
          current_confidence.mutable_data_ptr<float>(),
          rows,
          vocab_size);
      break;
    case at::kHalf:
      display_token_statistics_cpu_kernel<at::Half>(
          contiguous_logits.const_data_ptr<at::Half>(),
          contiguous_ids.const_data_ptr<std::int64_t>(),
          row_maxima.const_data_ptr<float>(),
          confidence.mutable_data_ptr<float>(),
          current_confidence.mutable_data_ptr<float>(),
          rows,
          vocab_size);
      break;
    case at::kBFloat16:
      display_token_statistics_cpu_kernel<at::BFloat16>(
          contiguous_logits.const_data_ptr<at::BFloat16>(),
          contiguous_ids.const_data_ptr<std::int64_t>(),
          row_maxima.const_data_ptr<float>(),
          confidence.mutable_data_ptr<float>(),
          current_confidence.mutable_data_ptr<float>(),
          rows,
          vocab_size);
      break;
    default:
      return display_token_statistics(token_ids, logits);
  }

  return {confidence, predicted, current_confidence};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CPU, m) {
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics_cpu));
}

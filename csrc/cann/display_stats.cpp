#include <ATen/Functions.h>
#include <torch/library.h>

namespace cid::engine {
namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> display_token_statistics_cann(
    const at::Tensor& token_ids,
    const at::Tensor& logits) {
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(
      logits.dim() == 3 &&
          logits.size(0) == token_ids.size(0) &&
          logits.size(1) == token_ids.size(1),
      "logits must have shape [batch, tokens, vocab]");
  TORCH_CHECK(logits.size(-1) > 0, "logits vocabulary dimension must be non-empty");

  auto probabilities = at::softmax(logits, -1, at::kFloat);
  at::Tensor confidence;
  at::Tensor predicted;

  if (logits.scalar_type() == at::kHalf ||
      logits.scalar_type() == at::kBFloat16) {
    auto maximum = probabilities.max(-1);
    confidence = std::get<0>(maximum);
    predicted = std::get<1>(maximum);
  } else {
    predicted = logits.argmax(-1);
    confidence = probabilities
                     .gather(-1, predicted.unsqueeze(-1))
                     .squeeze(-1);
  }

  auto current_confidence = probabilities
                                .gather(-1, token_ids.unsqueeze(-1))
                                .squeeze(-1);
  return {confidence, predicted, current_confidence};
}

}  // namespace
}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, PrivateUse1, m) {
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics_cann));
}

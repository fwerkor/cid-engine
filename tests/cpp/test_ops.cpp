#include "cid_engine/ops.hpp"

#include <ATen/Functions.h>

#include <iostream>
#include <vector>

int main() {
  auto options = at::TensorOptions().dtype(at::kFloat);
  auto occupancy = at::tensor(
      std::vector<float>{1.0F, 0.0F, 0.0F, 0.0F},
      options).reshape({1, 4, 1});
  auto logits = at::tensor(
      std::vector<float>{10.0F, 10.0F, 10.0F, -10.0F},
      options).reshape({1, 4});

  auto selected = cid::engine::prefix_allocation_mask(
      occupancy,
      logits,
      0.5,
      2);
  auto expected = at::zeros({1, 4}, at::TensorOptions().dtype(at::kBool));
  expected.index_put_({0, 1}, true);
  expected.index_put_({0, 2}, true);

  if (!selected.equal(expected)) {
    std::cerr << "prefix allocation mismatch\n";
    return 1;
  }

  auto lifecycle = at::zeros({1, 4, 3}, options);
  lifecycle.index_put_({0, 1, 2}, 1.0);
  auto live = cid::engine::live_slot_occupancy(
      occupancy,
      lifecycle,
      2);
  if (live.flatten().index({1}).item().toFloat() != 0.0F) {
    std::cerr << "retired occupancy mismatch\n";
    return 1;
  }

  std::cout << "cid-engine native C++ tests passed\n";
  return 0;
}

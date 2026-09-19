#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace cid::engine {
namespace {

constexpr std::int64_t kMaxColumns = 16;

__global__ void batched_linear_assignment_kernel(
    const float* __restrict__ costs,
    const std::int64_t* __restrict__ row_counts,
    std::int64_t* __restrict__ output,
    const std::int64_t rows,
    const std::int64_t columns) {
  if (threadIdx.x != 0) {
    return;
  }

  const std::int64_t batch_index = blockIdx.x;
  const std::int64_t active_rows = row_counts[batch_index];
  const std::int64_t output_offset = batch_index * rows;
  for (std::int64_t row = 0; row < rows; ++row) {
    output[output_offset + row] = -1;
  }
  if (active_rows <= 0 || active_rows > rows) {
    return;
  }

  double u[kMaxColumns + 1] = {};
  double v[kMaxColumns + 1] = {};
  std::int64_t p[kMaxColumns + 1] = {};
  std::int64_t way[kMaxColumns + 1] = {};
  const std::int64_t cost_offset = batch_index * rows * columns;

  for (std::int64_t row = 1; row <= active_rows; ++row) {
    p[0] = row;
    std::int64_t column0 = 0;
    double minimum[kMaxColumns + 1];
    bool used[kMaxColumns + 1] = {};
    for (std::int64_t column = 0; column <= columns; ++column) {
      minimum[column] = INFINITY;
    }

    do {
      used[column0] = true;
      const std::int64_t row0 = p[column0];
      double delta = INFINITY;
      std::int64_t column1 = 0;
      for (std::int64_t column = 1; column <= columns; ++column) {
        if (used[column]) {
          continue;
        }
        const std::int64_t index =
            cost_offset + (row0 - 1) * columns + (column - 1);
        const double current =
            static_cast<double>(costs[index]) - u[row0] - v[column];
        if (current < minimum[column]) {
          minimum[column] = current;
          way[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }

      for (std::int64_t column = 0; column <= columns; ++column) {
        if (used[column]) {
          u[p[column]] += delta;
          v[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (p[column0] != 0);

    do {
      const std::int64_t column1 = way[column0];
      p[column0] = p[column1];
      column0 = column1;
    } while (column0 != 0);
  }

  for (std::int64_t column = 1; column <= columns; ++column) {
    if (p[column] > 0 && p[column] <= active_rows) {
      output[output_offset + (p[column] - 1)] = column - 1;
    }
  }
}

}  // namespace

at::Tensor batched_linear_assignment_cuda(
    const at::Tensor& costs_input,
    const at::Tensor& row_counts_input) {
  TORCH_CHECK(costs_input.is_cuda(), "costs must be a CUDA tensor");
  TORCH_CHECK(row_counts_input.is_cuda(), "row_counts must be a CUDA tensor");
  TORCH_CHECK(
      costs_input.device() == row_counts_input.device(),
      "costs and row_counts must share a device");
  TORCH_CHECK(costs_input.dim() == 3, "costs must have shape [batch, rows, columns]");
  TORCH_CHECK(
      costs_input.scalar_type() == at::kFloat,
      "CUDA batched_linear_assignment requires float32 costs");
  TORCH_CHECK(
      row_counts_input.dim() == 1 &&
          row_counts_input.size(0) == costs_input.size(0),
      "row_counts must have shape [batch]");
  TORCH_CHECK(
      row_counts_input.scalar_type() == at::kLong,
      "row_counts must use torch.int64");
  TORCH_CHECK(
      costs_input.size(1) <= costs_input.size(2),
      "assignment rows cannot exceed columns");
  TORCH_CHECK(
      costs_input.size(2) <= kMaxColumns,
      "batched_linear_assignment supports at most 16 columns");

  const c10::cuda::CUDAGuard device_guard(costs_input.device());
  auto costs = costs_input.contiguous();
  auto row_counts = row_counts_input.contiguous();
  const auto batch = costs.size(0);
  const auto rows = costs.size(1);
  const auto columns = costs.size(2);
  auto output = at::empty(
      {batch, rows},
      costs.options().dtype(at::kLong));

  if (batch == 0 || rows == 0) {
    return output;
  }

  const auto stream = at::cuda::getCurrentCUDAStream(costs.get_device());
  batched_linear_assignment_kernel<<<batch, 1, 0, stream>>>(
      costs.const_data_ptr<float>(),
      row_counts.const_data_ptr<std::int64_t>(),
      output.mutable_data_ptr<std::int64_t>(),
      rows,
      columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "batched_linear_assignment",
      TORCH_FN(cid::engine::batched_linear_assignment_cuda));
}

#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr std::int64_t kMaxColumns = 16;
constexpr float kInfinity = 3.402823466e+38F;

}  // namespace

extern "C" __global__ __aicore__ void cid_batched_linear_assignment_kernel(
    GM_ADDR costs_address,
    GM_ADDR row_counts_address,
    GM_ADDR output_address,
    const std::int64_t batch,
    const std::int64_t rows,
    const std::int64_t columns) {
  AscendC::GlobalTensor<float> costs;
  AscendC::GlobalTensor<std::int64_t> row_counts;
  AscendC::GlobalTensor<std::int64_t> output;
  costs.SetGlobalBuffer((__gm__ float*)costs_address);
  row_counts.SetGlobalBuffer((__gm__ std::int64_t*)row_counts_address);
  output.SetGlobalBuffer((__gm__ std::int64_t*)output_address);

  const std::int64_t batch_index =
      static_cast<std::int64_t>(AscendC::GetBlockIdx());
  if (batch_index >= batch) {
    return;
  }

  const std::int64_t active_rows = row_counts.GetValue(batch_index);
  const std::int64_t output_offset = batch_index * rows;
  for (std::int64_t row = 0; row < rows; ++row) {
    output.SetValue(output_offset + row, -1);
  }
  if (active_rows <= 0 || active_rows > rows) {
    return;
  }

  float u[kMaxColumns + 1] = {};
  float v[kMaxColumns + 1] = {};
  std::int64_t p[kMaxColumns + 1] = {};
  std::int64_t way[kMaxColumns + 1] = {};
  const std::int64_t cost_offset = batch_index * rows * columns;

  for (std::int64_t row = 1; row <= active_rows; ++row) {
    p[0] = row;
    std::int64_t column0 = 0;
    float minimum[kMaxColumns + 1];
    bool used[kMaxColumns + 1] = {};
    for (std::int64_t column = 0; column <= columns; ++column) {
      minimum[column] = kInfinity;
    }

    do {
      used[column0] = true;
      const std::int64_t row0 = p[column0];
      float delta = kInfinity;
      std::int64_t column1 = 0;
      for (std::int64_t column = 1; column <= columns; ++column) {
        if (used[column]) {
          continue;
        }
        const std::int64_t index =
            cost_offset + (row0 - 1) * columns + (column - 1);
        const float current =
            costs.GetValue(index) - u[row0] - v[column];
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
      output.SetValue(output_offset + (p[column] - 1), column - 1);
    }
  }
}

extern "C" __global__ __aicore__ void cid_masked_diffusion_f32_kernel(
    GM_ADDR clean_ids_address,
    GM_ADDR ratio_random_address,
    GM_ADDR mask_random_address,
    GM_ADDR corrupted_address,
    GM_ADDR masked_address,
    GM_ADDR mask_ratio_address,
    GM_ADDR metrics_address,
    const std::int64_t batch,
    const std::int64_t tokens,
    const std::int64_t mask_token_id,
    const float min_mask_ratio,
    const float max_mask_ratio) {
  if (AscendC::GetBlockIdx() != 0) {
    return;
  }

  AscendC::GlobalTensor<std::int64_t> clean_ids;
  AscendC::GlobalTensor<float> ratio_random;
  AscendC::GlobalTensor<float> mask_random;
  AscendC::GlobalTensor<std::int64_t> corrupted;
  AscendC::GlobalTensor<bool> masked;
  AscendC::GlobalTensor<float> mask_ratio;
  AscendC::GlobalTensor<float> metrics;

  clean_ids.SetGlobalBuffer((__gm__ std::int64_t*)clean_ids_address);
  ratio_random.SetGlobalBuffer((__gm__ float*)ratio_random_address);
  mask_random.SetGlobalBuffer((__gm__ float*)mask_random_address);
  corrupted.SetGlobalBuffer((__gm__ std::int64_t*)corrupted_address);
  masked.SetGlobalBuffer((__gm__ bool*)masked_address);
  mask_ratio.SetGlobalBuffer((__gm__ float*)mask_ratio_address);
  metrics.SetGlobalBuffer((__gm__ float*)metrics_address);

  std::int64_t total_masked = 0;
  float ratio_sum = 0.0F;
  for (std::int64_t row = 0; row < batch; ++row) {
    const float ratio =
        min_mask_ratio +
        (max_mask_ratio - min_mask_ratio) * ratio_random.GetValue(row);
    mask_ratio.SetValue(row, ratio);
    ratio_sum += ratio;

    const std::int64_t base = row * tokens;
    std::int64_t selected_count = 0;
    float minimum = kInfinity;
    std::int64_t minimum_index = 0;

    for (std::int64_t column = 0; column < tokens; ++column) {
      const std::int64_t index = base + column;
      const float draw = mask_random.GetValue(index);
      const bool selected = draw < ratio;
      masked.SetValue(index, selected);
      corrupted.SetValue(
          index,
          selected ? mask_token_id : clean_ids.GetValue(index));
      selected_count += selected ? 1 : 0;
      if (draw < minimum) {
        minimum = draw;
        minimum_index = column;
      }
    }

    if (selected_count == 0) {
      const std::int64_t index = base + minimum_index;
      masked.SetValue(index, true);
      corrupted.SetValue(index, mask_token_id);
      selected_count = 1;
    }
    total_masked += selected_count;
  }

  metrics.SetValue(
      0,
      static_cast<float>(total_masked) /
          static_cast<float>(batch * tokens));
  metrics.SetValue(1, ratio_sum / static_cast<float>(batch));
}

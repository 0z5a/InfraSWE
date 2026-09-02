#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace ptx = cuda::ptx;

namespace {

constexpr int kWidth = 128;
constexpr int kHeight = 64;
constexpr int kRowsPerTransfer = 4;
constexpr int kElementsPerTransfer = kWidth * kRowsPerTransfer;
constexpr int kThreads = 128;
using block_barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(expr)                                                                  \
  do {                                                                                    \
    cudaError_t infraswe_cuda_status = (expr);                                            \
    if (infraswe_cuda_status != cudaSuccess) {                                            \
      std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__ << ": "             \
                << cudaGetErrorString(infraswe_cuda_status) << '\n';                      \
      std::exit(2);                                                                       \
    }                                                                                     \
  } while (false)

#define CU_CHECK(expr)                                                                    \
  do {                                                                                    \
    CUresult infraswe_cu_status = (expr);                                                  \
    if (infraswe_cu_status != CUDA_SUCCESS) {                                             \
      const char* infraswe_cu_name = nullptr;                                             \
      const char* infraswe_cu_message = nullptr;                                          \
      cuGetErrorName(infraswe_cu_status, &infraswe_cu_name);                              \
      cuGetErrorString(infraswe_cu_status, &infraswe_cu_message);                         \
      std::cerr << "CUDA driver failure at " << __FILE__ << ':' << __LINE__ << ": "      \
                << (infraswe_cu_name ? infraswe_cu_name : "unknown") << " ("             \
                << (infraswe_cu_message ? infraswe_cu_message : "unknown") << ")\n";   \
      std::exit(3);                                                                       \
    }                                                                                     \
  } while (false)

__device__ inline bool elected_lane() {
  const unsigned int warp_id = static_cast<unsigned int>(threadIdx.x) / 32U;
  const unsigned int uniform_warp_id = __shfl_sync(0xffffffffU, warp_id, 0);
  return uniform_warp_id == 0U && ptx::elect_sync(0xffffffffU);
}

__global__ void gather4_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int* output,
    int row0,
    int row1,
    int row2,
    int row3) {
  __shared__ alignas(128) int smem[kRowsPerTransfer][kWidth];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  block_barrier::arrival_token token;
  if (elected_lane()) {
    const int32_t coordinates[5] = {0, row0, row1, row2, row3};
    ptx::cp_async_bulk_tensor_tile_gather4(
        ptx::space_shared,
        ptx::space_global,
        &smem,
        &tensor_map,
        coordinates,
        cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  for (int index = static_cast<int>(threadIdx.x); index < kElementsPerTransfer;
       index += static_cast<int>(blockDim.x)) {
    output[index] = reinterpret_cast<int*>(smem)[index];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    (&bar)->~block_barrier();
  }
}

__global__ void scatter4_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const int* input,
    int row0,
    int row1,
    int row2,
    int row3) {
  __shared__ alignas(128) int smem[kRowsPerTransfer][kWidth];
  for (int index = static_cast<int>(threadIdx.x); index < kElementsPerTransfer;
       index += static_cast<int>(blockDim.x)) {
    reinterpret_cast<int*>(smem)[index] = input[index];
  }
  __syncthreads();
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  if (elected_lane()) {
    const int32_t coordinates[5] = {0, row0, row1, row2, row3};
    ptx::cp_async_bulk_tensor_tile_scatter4(
        ptx::space_global, ptx::space_shared, &tensor_map, coordinates, &smem);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}

__global__ void scalar_gather4_kernel(
    const int* input,
    int* output,
    int row0,
    int row1,
    int row2,
    int row3) {
  const int rows[4] = {row0, row1, row2, row3};
  for (int index = static_cast<int>(threadIdx.x); index < kElementsPerTransfer;
       index += static_cast<int>(blockDim.x)) {
    const int group_row = index / kWidth;
    const int column = index % kWidth;
    output[index] = input[rows[group_row] * kWidth + column];
  }
}

__global__ void scalar_scatter4_kernel(
    const int* input,
    int* output,
    int row0,
    int row1,
    int row2,
    int row3) {
  const int rows[4] = {row0, row1, row2, row3};
  for (int index = static_cast<int>(threadIdx.x); index < kElementsPerTransfer;
       index += static_cast<int>(blockDim.x)) {
    const int group_row = index / kWidth;
    const int column = index % kWidth;
    output[rows[group_row] * kWidth + column] = input[index];
  }
}

CUtensorMap encode_tensor_map(int* device_pointer) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  const uint64_t global_dimensions[rank] = {kWidth, kHeight};
  const uint64_t global_strides[rank - 1] = {kWidth * sizeof(int)};
  // gather4/scatter4 form four independent bounding boxes.  PTX requires the
  // bounding box in dimension 1 to be exactly one row; the instruction itself
  // supplies the four row coordinates and packs/unpacks those four rows.
  const uint32_t box_dimensions[rank] = {kWidth, 1};
  const uint32_t element_strides[rank] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,
      device_pointer,
      global_dimensions,
      global_strides,
      box_dimensions,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

template <typename Launch>
float benchmark_us(Launch&& launch, int warmups, int iterations) {
  for (int index = 0; index < warmups; ++index) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int index = 0; index < iterations; ++index) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms * 1000.0F / static_cast<float>(iterations);
}

bool verify_gather(
    const std::vector<int>& source,
    const std::vector<int>& output,
    const std::array<int, 4>& rows) {
  for (int group_row = 0; group_row < 4; ++group_row) {
    for (int column = 0; column < kWidth; ++column) {
      if (output[group_row * kWidth + column] != source[rows[group_row] * kWidth + column]) {
        return false;
      }
    }
  }
  return true;
}

bool verify_scatter(
    const std::vector<int>& input,
    const std::vector<int>& output,
    const std::array<int, 4>& rows,
    int sentinel) {
  for (int row = 0; row < kHeight; ++row) {
    const bool selected = std::find(rows.begin(), rows.end(), row) != rows.end();
    for (int column = 0; column < kWidth; ++column) {
      const int expected = selected
                               ? input[static_cast<int>(
                                     std::find(rows.begin(), rows.end(), row) - rows.begin()) *
                                         kWidth +
                                     column]
                               : sentinel;
      if (output[row * kWidth + column] != expected) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int iterations = 5000;
  if (argc > 1) {
    iterations = std::max(100, std::atoi(argv[1]));
  }
  CUDA_CHECK(cudaSetDevice(0));
  CU_CHECK(cuInit(0));

  std::vector<int> host_source(kWidth * kHeight);
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      host_source[row * kWidth + column] = row * 100000 + column * 7 + 3;
    }
  }
  std::vector<int> host_transfer(kElementsPerTransfer);
  for (int index = 0; index < kElementsPerTransfer; ++index) {
    host_transfer[index] = 9000000 + index * 13;
  }

  int* device_source = nullptr;
  int* device_gather = nullptr;
  int* device_scatter = nullptr;
  CUDA_CHECK(cudaMalloc(&device_source, host_source.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_gather, host_transfer.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_scatter, host_source.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(
      device_source,
      host_source.data(),
      host_source.size() * sizeof(int),
      cudaMemcpyHostToDevice));

  const CUtensorMap source_map = encode_tensor_map(device_source);
  const CUtensorMap scatter_map = encode_tensor_map(device_scatter);
  const std::array<std::array<int, 4>, 4> gather_cases = {
      std::array<int, 4>{1, 7, 19, 41},
      std::array<int, 4>{3, 3, 27, 2},
      std::array<int, 4>{63, 40, 11, 0},
      std::array<int, 4>{9, 8, 7, 6},
  };
  bool gather_correct = true;
  std::vector<int> host_gather(kElementsPerTransfer);
  for (const auto& rows : gather_cases) {
    gather4_kernel<<<1, kThreads>>>(
        source_map, device_gather, rows[0], rows[1], rows[2], rows[3]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        host_gather.data(),
        device_gather,
        host_gather.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    gather_correct = gather_correct && verify_gather(host_source, host_gather, rows);
  }

  const std::array<int, 4> scatter_rows = {2, 17, 33, 61};
  constexpr int sentinel = -7654321;
  CUDA_CHECK(cudaMemcpy(
      device_gather,
      host_transfer.data(),
      host_transfer.size() * sizeof(int),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_scatter, 0, host_source.size() * sizeof(int)));
  std::vector<int> sentinel_buffer(host_source.size(), sentinel);
  CUDA_CHECK(cudaMemcpy(
      device_scatter,
      sentinel_buffer.data(),
      sentinel_buffer.size() * sizeof(int),
      cudaMemcpyHostToDevice));
  scatter4_kernel<<<1, kThreads>>>(
      scatter_map,
      device_gather,
      scatter_rows[0],
      scatter_rows[1],
      scatter_rows[2],
      scatter_rows[3]);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> host_scatter(host_source.size());
  CUDA_CHECK(cudaMemcpy(
      host_scatter.data(),
      device_scatter,
      host_scatter.size() * sizeof(int),
      cudaMemcpyDeviceToHost));
  const bool scatter_correct =
      verify_scatter(host_transfer, host_scatter, scatter_rows, sentinel);

  const auto perf_rows = gather_cases[0];
  const float gather4_us = benchmark_us(
      [&] {
        gather4_kernel<<<1, kThreads>>>(source_map,
                                        device_gather,
                                        perf_rows[0],
                                        perf_rows[1],
                                        perf_rows[2],
                                        perf_rows[3]);
      },
      100,
      iterations);
  const float scalar_gather_us = benchmark_us(
      [&] {
        scalar_gather4_kernel<<<1, kThreads>>>(device_source,
                                               device_gather,
                                               perf_rows[0],
                                               perf_rows[1],
                                               perf_rows[2],
                                               perf_rows[3]);
      },
      100,
      iterations);
  const float scatter4_us = benchmark_us(
      [&] {
        scatter4_kernel<<<1, kThreads>>>(scatter_map,
                                         device_gather,
                                         scatter_rows[0],
                                         scatter_rows[1],
                                         scatter_rows[2],
                                         scatter_rows[3]);
      },
      100,
      iterations);
  const float scalar_scatter_us = benchmark_us(
      [&] {
        scalar_scatter4_kernel<<<1, kThreads>>>(device_gather,
                                                device_scatter,
                                                scatter_rows[0],
                                                scatter_rows[1],
                                                scatter_rows[2],
                                                scatter_rows[3]);
      },
      100,
      iterations);

  const bool passed = gather_correct && scatter_correct;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"schema_version\":\"0.1\",\"feature_id\":\"BW-TMA-001\","
            << "\"status\":\"" << (passed ? "passed" : "failed") << "\","
            << "\"shape\":[" << kHeight << ',' << kWidth << "],"
            << "\"bytes_per_transfer\":" << kElementsPerTransfer * sizeof(int) << ','
            << "\"correctness\":{\"gather4\":" << (gather_correct ? "true" : "false")
            << ",\"scatter4\":" << (scatter_correct ? "true" : "false")
            << ",\"case_count\":" << gather_cases.size() + 1 << "},"
            << "\"performance\":{\"iterations\":" << iterations
            << ",\"gather4_us\":" << gather4_us
            << ",\"scalar_gather4_us\":" << scalar_gather_us
            << ",\"scatter4_us\":" << scatter4_us
            << ",\"scalar_scatter4_us\":" << scalar_scatter_us << "}}\n";

  CUDA_CHECK(cudaFree(device_source));
  CUDA_CHECK(cudaFree(device_gather));
  CUDA_CHECK(cudaFree(device_scatter));
  return passed ? 0 : 1;
}

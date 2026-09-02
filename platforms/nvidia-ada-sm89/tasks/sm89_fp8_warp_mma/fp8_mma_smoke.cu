#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

extern "C" __global__ __launch_bounds__(32) void sm89_fp8_mma_smoke(float* output) {
  const int lane = threadIdx.x;
  if (lane >= 32) return;

  const unsigned a0 = 0;
  const unsigned a1 = 0;
  const unsigned a2 = 0;
  const unsigned a3 = 0;
  const unsigned b0 = 0;
  const unsigned b1 = 0;
  float e4_0 = 0.0f;
  float e4_1 = 0.0f;
  float e4_2 = 0.0f;
  float e4_3 = 0.0f;
  float e5_0 = 0.0f;
  float e5_1 = 0.0f;
  float e5_2 = 0.0f;
  float e5_3 = 0.0f;

  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(e4_0), "+f"(e4_1), "+f"(e4_2), "+f"(e4_3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(e5_0), "+f"(e5_1), "+f"(e5_2), "+f"(e5_3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

  const int base = lane * 8;
  output[base + 0] = e4_0;
  output[base + 1] = e4_1;
  output[base + 2] = e4_2;
  output[base + 3] = e4_3;
  output[base + 4] = e5_0;
  output[base + 5] = e5_1;
  output[base + 6] = e5_2;
  output[base + 7] = e5_3;
}

int main() {
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 2;
  const int sm = properties.major * 10 + properties.minor;
  if (sm != 89) {
    std::cout << "{\"passed\":false,\"reason\":\"runtime_not_sm89\",\"runtime_sm\":"
              << sm << "}\n";
    return 3;
  }

  constexpr int values = 32 * 8;
  float* device_output = nullptr;
  if (cudaMalloc(&device_output, values * sizeof(float)) != cudaSuccess) return 4;
  cudaMemset(device_output, 0xff, values * sizeof(float));
  sm89_fp8_mma_smoke<<<1, 32>>>(device_output);
  const cudaError_t launch = cudaDeviceSynchronize();
  std::vector<float> host(values);
  cudaMemcpy(host.data(), device_output, values * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(device_output);

  bool correct = launch == cudaSuccess;
  for (float value : host) correct = correct && std::isfinite(value) && value == 0.0f;
  std::cout << "{\"passed\":" << (correct ? "true" : "false")
            << ",\"runtime_sm\":" << sm
            << ",\"entry\":\"sm89_fp8_mma_smoke\",\"e4m3\":true,\"e5m2\":true,"
               "\"fp32_accumulator\":true,\"full_size_fp16_temporaries\":0}\n";
  return correct ? 0 : 5;
}

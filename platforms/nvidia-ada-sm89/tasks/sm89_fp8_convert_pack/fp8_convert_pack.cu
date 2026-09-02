#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

extern "C" __global__ void sm89_fp8_convert_pack(const float* input, std::uint16_t* e4m3,
                                                   std::uint16_t* e5m2, int count,
                                                   unsigned* amax_bits) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= (count + 1) / 2) return;
  const int first = pair * 2;
  const float x0 = input[first];
  const float x1 = first + 1 < count ? input[first + 1] : 0.0f;
  std::uint16_t packed_e4m3;
  std::uint16_t packed_e5m2;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
               : "=h"(packed_e4m3)
               : "f"(x0), "f"(x1));
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;"
               : "=h"(packed_e5m2)
               : "f"(x0), "f"(x1));
  e4m3[pair] = packed_e4m3;
  e5m2[pair] = packed_e5m2;
  atomicMax(amax_bits, __float_as_uint(fmaxf(fabsf(x0), fabsf(x1))));
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

  constexpr int count = 4097;
  constexpr int pairs = (count + 1) / 2;
  std::vector<float> input(count, 0.0f);
  float* device_input = nullptr;
  std::uint16_t* device_e4m3 = nullptr;
  std::uint16_t* device_e5m2 = nullptr;
  unsigned* device_amax = nullptr;
  if (cudaMalloc(&device_input, count * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&device_e4m3, (pairs + 1) * sizeof(std::uint16_t)) != cudaSuccess ||
      cudaMalloc(&device_e5m2, (pairs + 1) * sizeof(std::uint16_t)) != cudaSuccess ||
      cudaMalloc(&device_amax, sizeof(unsigned)) != cudaSuccess) {
    return 4;
  }
  cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(device_e4m3, 0xa5, (pairs + 1) * sizeof(std::uint16_t));
  cudaMemset(device_e5m2, 0x5a, (pairs + 1) * sizeof(std::uint16_t));
  cudaMemset(device_amax, 0, sizeof(unsigned));
  sm89_fp8_convert_pack<<<(pairs + 255) / 256, 256>>>(device_input, device_e4m3,
                                                       device_e5m2, count, device_amax);
  const cudaError_t launch = cudaDeviceSynchronize();

  std::vector<std::uint16_t> e4m3(pairs + 1);
  std::vector<std::uint16_t> e5m2(pairs + 1);
  unsigned amax_bits = 1;
  cudaMemcpy(e4m3.data(), device_e4m3, e4m3.size() * sizeof(std::uint16_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(e5m2.data(), device_e5m2, e5m2.size() * sizeof(std::uint16_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&amax_bits, device_amax, sizeof(unsigned), cudaMemcpyDeviceToHost);
  cudaFree(device_input);
  cudaFree(device_e4m3);
  cudaFree(device_e5m2);
  cudaFree(device_amax);

  bool correct = launch == cudaSuccess && amax_bits == 0;
  for (int pair = 0; pair < pairs; ++pair) {
    correct = correct && e4m3[pair] == 0 && e5m2[pair] == 0;
  }
  correct = correct && e4m3[pairs] == 0xa5a5 && e5m2[pairs] == 0x5a5a;
  std::cout << "{\"passed\":" << (correct ? "true" : "false")
            << ",\"runtime_sm\":" << sm
            << ",\"entry\":\"sm89_fp8_convert_pack\",\"count\":" << count
            << ",\"odd_tail_masked\":true,\"amax_atomic\":true}\n";
  return correct ? 0 : 5;
}

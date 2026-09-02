#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

extern "C" __global__ void sm89_cpasync_pipeline(const std::uint8_t* input,
                                                   std::uint8_t* output, int count) {
  extern __shared__ std::uint8_t staging[];
  const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (offset >= count) return;
  const int remaining = count - offset;
  const int valid_bytes = remaining < 16 ? remaining : 16;
  const unsigned shared_address = static_cast<unsigned>(
      __cvta_generic_to_shared(staging + static_cast<int>(threadIdx.x) * 16));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :
               : "r"(shared_address), "l"(input + offset), "r"(valid_bytes)
               : "memory");
  asm volatile("cp.async.commit_group;" : : : "memory");
  asm volatile("cp.async.wait_group 0;" : : : "memory");
  const std::uint8_t* local = staging + static_cast<int>(threadIdx.x) * 16;
#pragma unroll
  for (int byte = 0; byte < 16; ++byte) {
    if (byte < valid_bytes) output[offset + byte] = local[byte];
  }
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

  constexpr int count = 4099;
  constexpr int threads = 256;
  const int copies = (count + 15) / 16;
  const int blocks = (copies + threads - 1) / threads;
  std::vector<std::uint8_t> input(count);
  for (int index = 0; index < count; ++index) input[index] = (index * 17 + 5) & 0xff;
  std::uint8_t* device_input = nullptr;
  std::uint8_t* device_output = nullptr;
  if (cudaMalloc(&device_input, count) != cudaSuccess ||
      cudaMalloc(&device_output, count) != cudaSuccess) {
    return 4;
  }
  cudaMemcpy(device_input, input.data(), count, cudaMemcpyHostToDevice);
  cudaMemset(device_output, 0, count);
  sm89_cpasync_pipeline<<<blocks, threads, threads * 16>>>(device_input, device_output, count);
  const cudaError_t launch = cudaDeviceSynchronize();
  std::vector<std::uint8_t> output(count);
  cudaMemcpy(output.data(), device_output, count, cudaMemcpyDeviceToHost);
  cudaFree(device_input);
  cudaFree(device_output);

  const bool correct = launch == cudaSuccess && output == input;
  std::cout << "{\"passed\":" << (correct ? "true" : "false")
            << ",\"runtime_sm\":" << sm
            << ",\"entry\":\"sm89_cpasync_pipeline\",\"count\":" << count
            << ",\"tail_zero_fill\":true,\"copy_bytes\":16}\n";
  return correct ? 0 : 5;
}

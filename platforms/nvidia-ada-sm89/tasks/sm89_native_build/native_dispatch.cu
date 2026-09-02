#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" __global__ void sm89_native_dispatch_kernel(int* values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] = values[index] * 3 + 1;
}

int main() {
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 2;
  const int sm = properties.major * 10 + properties.minor;
  const bool force_ptx_jit = std::getenv("CUDA_FORCE_PTX_JIT") != nullptr;
  if (sm != 89) {
    std::cout << "{\"passed\":false,\"reason\":\"runtime_not_sm89\",\"runtime_sm\":"
              << sm << ",\"dispatch_mode\":\""
              << (force_ptx_jit ? "ptx_jit" : "native_cubin") << "\"}\n";
    return 3;
  }

  constexpr int count = 4096;
  std::vector<int> host(count);
  for (int index = 0; index < count; ++index) host[index] = index;
  int* device_values = nullptr;
  if (cudaMalloc(&device_values, count * sizeof(int)) != cudaSuccess) return 4;
  cudaMemcpy(device_values, host.data(), count * sizeof(int), cudaMemcpyHostToDevice);
  sm89_native_dispatch_kernel<<<(count + 255) / 256, 256>>>(device_values, count);
  const cudaError_t launch = cudaDeviceSynchronize();
  cudaMemcpy(host.data(), device_values, count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_values);

  bool correct = launch == cudaSuccess;
  for (int index = 0; index < count && correct; ++index) {
    correct = host[index] == index * 3 + 1;
  }
  std::cout << "{\"passed\":" << (correct ? "true" : "false")
            << ",\"runtime_sm\":" << sm
            << ",\"dispatch_mode\":\"" << (force_ptx_jit ? "ptx_jit" : "native_cubin")
            << "\",\"entry\":\"sm89_native_dispatch_kernel\",\"count\":" << count
            << "}\n";
  return correct ? 0 : 5;
}

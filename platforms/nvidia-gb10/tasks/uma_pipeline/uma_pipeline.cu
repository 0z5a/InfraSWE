#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void increment_kernel(int* values, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] += 7;
}

bool verify(const int* values, int count) {
  for (int index = 0; index < count; ++index) {
    if (values[index] != index + 7) return false;
  }
  return true;
}

int main() {
  constexpr int count = 1 << 20;
  constexpr std::size_t bytes = count * sizeof(int);
  int pageable = 0;
  int host_tables = 0;
  int unified = 0;
  cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
  cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, 0);

  int* system_values = static_cast<int*>(std::malloc(bytes));
  if (!system_values) return 2;
  for (int index = 0; index < count; ++index) system_values[index] = index;
  increment_kernel<<<(count + 255) / 256, 256>>>(system_values, count);
  cudaError_t system_status = cudaDeviceSynchronize();
  bool system_passed = system_status == cudaSuccess && verify(system_values, count);
  std::free(system_values);

  int* managed_values = nullptr;
  cudaError_t managed_alloc = cudaMallocManaged(&managed_values, bytes);
  bool managed_passed = managed_alloc == cudaSuccess;
  if (managed_passed) {
    for (int index = 0; index < count; ++index) managed_values[index] = index;
    increment_kernel<<<(count + 255) / 256, 256>>>(managed_values, count);
    managed_passed = cudaDeviceSynchronize() == cudaSuccess && verify(managed_values, count);
    cudaFree(managed_values);
  }

  int* pinned_values = nullptr;
  cudaError_t pinned_alloc = cudaHostAlloc(&pinned_values, bytes, cudaHostAllocMapped);
  bool pinned_passed = pinned_alloc == cudaSuccess;
  if (pinned_passed) {
    for (int index = 0; index < count; ++index) pinned_values[index] = index;
    int* mapped = nullptr;
    pinned_passed = cudaHostGetDevicePointer(&mapped, pinned_values, 0) == cudaSuccess;
    if (pinned_passed) {
      increment_kernel<<<(count + 255) / 256, 256>>>(mapped, count);
      pinned_passed = cudaDeviceSynchronize() == cudaSuccess && verify(pinned_values, count);
    }
    cudaFreeHost(pinned_values);
  }

  int* private_values = nullptr;
  std::vector<int> private_host(count);
  for (int index = 0; index < count; ++index) private_host[index] = index;
  bool private_passed = cudaMalloc(&private_values, bytes) == cudaSuccess;
  if (private_passed) {
    cudaMemcpy(private_values, private_host.data(), bytes, cudaMemcpyHostToDevice);
    increment_kernel<<<(count + 255) / 256, 256>>>(private_values, count);
    private_passed = cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(private_host.data(), private_values, bytes, cudaMemcpyDeviceToHost);
    private_passed = private_passed && verify(private_host.data(), count);
    cudaFree(private_values);
  }

  bool passed = pageable == 1 && host_tables == 1 && unified == 1 && system_passed &&
                managed_passed && pinned_passed && private_passed;
  std::cout << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"pageable_memory_access\":" << pageable
            << ",\"host_page_table_coherence\":" << host_tables
            << ",\"unified_addressing\":" << unified
            << ",\"system_allocation\":" << (system_passed ? "true" : "false")
            << ",\"managed_allocation\":" << (managed_passed ? "true" : "false")
            << ",\"pinned_allocation\":" << (pinned_passed ? "true" : "false")
            << ",\"gpu_private_via_copy\":" << (private_passed ? "true" : "false")
            << ",\"cpu_dereferenced_cuda_malloc\":false,\"bytes_per_path\":" << bytes
            << "}\n";
  return passed ? 0 : 6;
}

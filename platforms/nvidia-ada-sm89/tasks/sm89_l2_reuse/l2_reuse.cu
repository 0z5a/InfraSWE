#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

extern "C" __global__ void sm89_l2_reuse_kernel(const float* weights, float* output,
                                                  int elements, int rounds) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int round = 0; round < rounds; ++round) {
    for (int index = thread; index < elements; index += gridDim.x * blockDim.x) {
      sum += weights[index];
    }
  }
  if (thread < 256) output[thread] = sum;
}

float run_once(cudaStream_t stream, const float* weights, float* output, int elements,
               int rounds) {
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  sm89_l2_reuse_kernel<<<128, 256, 0, stream>>>(weights, output, elements, rounds);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
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

  const std::size_t target_bytes = std::min<std::size_t>(
      static_cast<std::size_t>(properties.l2CacheSize) / 4, 16ULL * 1024 * 1024);
  const int elements = static_cast<int>(std::max<std::size_t>(target_bytes / sizeof(float), 1024));
  const std::size_t allocation_bytes = static_cast<std::size_t>(elements) * sizeof(float);
  std::vector<float> host_weights(elements, 1.0f);
  float* weights = nullptr;
  float* output = nullptr;
  if (cudaMalloc(&weights, allocation_bytes) != cudaSuccess ||
      cudaMalloc(&output, 256 * sizeof(float)) != cudaSuccess) {
    return 4;
  }
  cudaMemcpy(weights, host_weights.data(), allocation_bytes, cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const std::size_t persist_limit = std::min<std::size_t>(
      static_cast<std::size_t>(properties.persistingL2CacheMaxSize),
      static_cast<std::size_t>(properties.l2CacheSize) / 2);
  cudaError_t limit_result = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_limit);
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr = weights;
  attribute.accessPolicyWindow.num_bytes = std::min<std::size_t>(
      allocation_bytes, static_cast<std::size_t>(properties.accessPolicyMaxWindowSize));
  attribute.accessPolicyWindow.hitRatio = 1.0;
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaError_t policy_result = cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attribute);

  constexpr int rounds = 8;
  const float cold_ms = run_once(stream, weights, output, elements, rounds);
  const float warm_ms = run_once(stream, weights, output, elements, rounds);
  std::vector<float> host_output(256);
  cudaMemcpy(host_output.data(), output, host_output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  bool finite = true;
  for (float value : host_output) finite = finite && std::isfinite(value) && value > 0.0f;

  attribute.accessPolicyWindow.num_bytes = 0;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attribute);
  cudaCtxResetPersistingL2Cache();
  cudaStreamDestroy(stream);
  cudaFree(weights);
  cudaFree(output);

  const bool correct = finite && limit_result == cudaSuccess && policy_result == cudaSuccess;
  std::cout << "{\"passed\":" << (correct ? "true" : "false")
            << ",\"runtime_sm\":" << sm
            << ",\"entry\":\"sm89_l2_reuse_kernel\",\"l2_bytes\":"
            << properties.l2CacheSize << ",\"working_set_bytes\":" << allocation_bytes
            << ",\"persisting_limit_bytes\":" << persist_limit
            << ",\"cold_ms\":" << cold_ms << ",\"warm_ms\":" << warm_ms
            << ",\"allocation_copies\":1}\n";
  return correct ? 0 : 5;
}

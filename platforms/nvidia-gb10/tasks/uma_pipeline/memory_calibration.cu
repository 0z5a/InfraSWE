#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__global__ void copy_kernel(const float* input, float* output, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index];
}

__global__ void empty_kernel() {}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

int run(int argc, char** argv) {
  std::size_t bytes = 256ULL << 20;
  int iterations = 40;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (index + 1 >= argc) throw std::runtime_error("missing calibration argument");
    if (key == "--bytes") bytes = std::stoull(argv[++index]);
    else if (key == "--iterations") iterations = std::stoi(argv[++index]);
    else throw std::runtime_error("unknown calibration argument: " + key);
  }
  bytes = (bytes / sizeof(float)) * sizeof(float);
  if (bytes < 4096 || iterations < 5) throw std::runtime_error("invalid calibration size");
  const std::size_t count = bytes / sizeof(float);
  float* input = static_cast<float*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096));
  float* output = static_cast<float*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096));
  if (!input || !output) throw std::bad_alloc();
  for (std::size_t index = 0; index < count; ++index) input[index] = index % 251;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check(cudaEventCreate(&start), "cudaEventCreate start");
  check(cudaEventCreate(&stop), "cudaEventCreate stop");
  copy_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(input, output, count);
  check(cudaDeviceSynchronize(), "copy warmup");

  std::vector<float> copy_ms;
  std::vector<float> launch_us;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check(cudaEventRecord(start), "copy start");
    copy_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(input, output, count);
    check(cudaEventRecord(stop), "copy stop");
    check(cudaEventSynchronize(stop), "copy synchronize");
    float elapsed_ms = 0;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "copy elapsed");
    copy_ms.push_back(elapsed_ms);

    check(cudaEventRecord(start), "launch start");
    empty_kernel<<<1, 1>>>();
    check(cudaEventRecord(stop), "launch stop");
    check(cudaEventSynchronize(stop), "launch synchronize");
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "launch elapsed");
    launch_us.push_back(elapsed_ms * 1000.0f);
  }
  const double copy_milliseconds = median(copy_ms);
  const double launch_microseconds = median(launch_us);
  const double bandwidth_gbps = (2.0 * static_cast<double>(bytes)) / (copy_milliseconds * 1e6);
  const std::size_t probes[] = {0, count / 2, count - 1};
  bool passed = true;
  for (const std::size_t index : probes) {
    if (std::abs(output[index] - input[index]) > 1e-6f) passed = false;
  }
  std::cout << std::setprecision(12) << "{\"schema_version\":\"0.4\",\"passed\":"
            << (passed ? "true" : "false") << ",\"bytes_per_buffer\":" << bytes
            << ",\"iterations\":" << iterations << ",\"median_copy_ms\":"
            << copy_milliseconds << ",\"memory_bandwidth_gbps\":" << bandwidth_gbps
            << ",\"launch_floor_us\":" << launch_microseconds
            << ",\"calibration_memory\":\"pageable-system-direct\"}\n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::free(input);
  std::free(output);
  return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cout << "{\"schema_version\":\"0.4\",\"passed\":false,\"failure_code\":"
                 "\"GB10_CALIBRATION_FAILED\",\"reason\":\""
              << error.what() << "\"}\n";
    return 3;
  }
}

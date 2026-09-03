#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace {

[[noreturn]] void fail_driver(const char* step, CUresult result) {
  const char* name = "unknown";
  const char* message = "unknown";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::printf(
      "{\"status\":\"failed\",\"failed_step\":\"%s\","
      "\"returncode\":%d,\"error_name\":\"%s\","
      "\"error_string\":\"%s\"}\n",
      step, static_cast<int>(result), name, message);
  std::exit(2);
}

[[noreturn]] void fail_runtime(const char* step, cudaError_t result) {
  std::printf(
      "{\"status\":\"failed\",\"failed_step\":\"%s\","
      "\"returncode\":%d,\"error_name\":\"%s\","
      "\"error_string\":\"%s\"}\n",
      step, static_cast<int>(result), cudaGetErrorName(result),
      cudaGetErrorString(result));
  std::exit(3);
}

#define DRIVER_CHECK(step, call)          \
  do {                                    \
    const CUresult result = (call);        \
    if (result != CUDA_SUCCESS) {          \
      fail_driver((step), result);         \
    }                                     \
  } while (false)

#define RUNTIME_CHECK(step, call)          \
  do {                                     \
    const cudaError_t result = (call);      \
    if (result != cudaSuccess) {            \
      fail_runtime((step), result);         \
    }                                      \
  } while (false)

size_t round_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__global__ void fill_words(std::uint32_t* destination, std::uint32_t value,
                           size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    destination[index] = value;
  }
}

__global__ void multimem_load_reduce_probe(const std::uint32_t* multicast,
                                           std::uint32_t* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  std::uint32_t value;
  asm volatile("fence.proxy.alias;" ::: "memory");
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
      : "=r"(value)
      : "l"(multicast)
      : "memory");
  *output = value;
#else
  *output = 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 7;
  const std::uint32_t second = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 11;
  constexpr int device_count = 2;
  constexpr int iterations = 1000;
  constexpr size_t requested_size = 1U << 20;

  DRIVER_CHECK("cuInit", cuInit(0));
  int visible_devices = 0;
  DRIVER_CHECK("cuDeviceGetCount", cuDeviceGetCount(&visible_devices));
  if (visible_devices < device_count) {
    std::printf(
        "{\"status\":\"skipped\",\"reason\":\"requires-two-devices\","
        "\"visible_device_count\":%d}\n",
        visible_devices);
    return 4;
  }

  CUdevice devices[device_count]{};
  int multicast_attributes[device_count]{};
  for (int index = 0; index < device_count; ++index) {
    DRIVER_CHECK("cuDeviceGet", cuDeviceGet(&devices[index], index));
    DRIVER_CHECK(
        "cuDeviceGetAttribute.multicast",
        cuDeviceGetAttribute(&multicast_attributes[index],
                             CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                             devices[index]));
    if (multicast_attributes[index] != 1) {
      std::printf(
          "{\"status\":\"skipped\","
          "\"reason\":\"multicast-attribute-disabled\","
          "\"device\":%d}\n",
          index);
      return 4;
    }
    RUNTIME_CHECK("cudaSetDevice.initialize", cudaSetDevice(index));
    RUNTIME_CHECK("cudaFree.initialize", cudaFree(nullptr));
  }

  CUmulticastObjectProp multicast_properties{};
  multicast_properties.numDevices = device_count;
  multicast_properties.size = requested_size;
  multicast_properties.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t multicast_minimum = 0;
  size_t multicast_recommended = 0;
  DRIVER_CHECK(
      "cuMulticastGetGranularity.minimum",
      cuMulticastGetGranularity(&multicast_minimum, &multicast_properties,
                                CU_MULTICAST_GRANULARITY_MINIMUM));
  DRIVER_CHECK(
      "cuMulticastGetGranularity.recommended",
      cuMulticastGetGranularity(&multicast_recommended, &multicast_properties,
                                CU_MULTICAST_GRANULARITY_RECOMMENDED));

  CUmemAllocationProp allocation_properties[device_count]{};
  size_t allocation_granularities[device_count]{};
  size_t common_alignment = multicast_recommended;
  for (int index = 0; index < device_count; ++index) {
    auto& properties = allocation_properties[index];
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = devices[index];
    DRIVER_CHECK(
        "cuMemGetAllocationGranularity.recommended",
        cuMemGetAllocationGranularity(
            &allocation_granularities[index], &properties,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    common_alignment = std::lcm(common_alignment, allocation_granularities[index]);
  }
  const size_t allocation_size = round_up(requested_size, common_alignment);
  multicast_properties.size = allocation_size;

  RUNTIME_CHECK("cudaSetDevice.create", cudaSetDevice(0));
  CUmemGenericAllocationHandle multicast_handle{};
  DRIVER_CHECK("cuMulticastCreate",
               cuMulticastCreate(&multicast_handle, &multicast_properties));
  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.add", cudaSetDevice(index));
    DRIVER_CHECK("cuMulticastAddDevice",
                 cuMulticastAddDevice(multicast_handle, devices[index]));
  }

  CUmemGenericAllocationHandle unicast_handles[device_count]{};
  CUdeviceptr unicast_pointers[device_count]{};
  const std::uint32_t inputs[device_count] = {first, second};
  const size_t word_count = allocation_size / sizeof(std::uint32_t);
  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.allocate", cudaSetDevice(index));
    DRIVER_CHECK(
        "cuMemAddressReserve.unicast",
        cuMemAddressReserve(&unicast_pointers[index], allocation_size,
                            common_alignment, 0, 0));
    DRIVER_CHECK("cuMemCreate",
                 cuMemCreate(&unicast_handles[index], allocation_size,
                             &allocation_properties[index], 0));
    DRIVER_CHECK("cuMemMap.unicast",
                 cuMemMap(unicast_pointers[index], allocation_size, 0,
                          unicast_handles[index], 0));
    CUmemAccessDesc access{};
    access.location = allocation_properties[index].location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRIVER_CHECK("cuMemSetAccess.unicast",
                 cuMemSetAccess(unicast_pointers[index], allocation_size,
                                &access, 1));
    fill_words<<<static_cast<unsigned>((word_count + 255) / 256), 256>>>(
        reinterpret_cast<std::uint32_t*>(unicast_pointers[index]),
        inputs[index], word_count);
    RUNTIME_CHECK("fill_words.launch", cudaGetLastError());
    RUNTIME_CHECK("fill_words.synchronize", cudaDeviceSynchronize());
  }

  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.bind", cudaSetDevice(index));
    DRIVER_CHECK("cuMulticastBindMem",
                 cuMulticastBindMem(multicast_handle, 0,
                                    unicast_handles[index], 0,
                                    allocation_size, 0));
  }

  RUNTIME_CHECK("cudaSetDevice.map_multicast", cudaSetDevice(0));
  CUdeviceptr multicast_pointer{};
  DRIVER_CHECK(
      "cuMemAddressReserve.multicast",
      cuMemAddressReserve(&multicast_pointer, allocation_size,
                          multicast_recommended, 0, 0));
  DRIVER_CHECK("cuMemMap.multicast",
               cuMemMap(multicast_pointer, allocation_size, 0,
                        multicast_handle, 0));
  CUmemAccessDesc multicast_access[device_count]{};
  for (int index = 0; index < device_count; ++index) {
    multicast_access[index].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    multicast_access[index].location.id = devices[index];
    multicast_access[index].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  DRIVER_CHECK("cuMemSetAccess.multicast",
               cuMemSetAccess(multicast_pointer, allocation_size,
                              multicast_access, device_count));

  std::uint32_t observed[device_count]{};
  float latency_us[device_count]{};
  std::uint32_t* outputs[device_count]{};
  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.execute", cudaSetDevice(index));
    RUNTIME_CHECK("cudaMalloc.output",
                  cudaMalloc(reinterpret_cast<void**>(&outputs[index]),
                             sizeof(std::uint32_t)));
    multimem_load_reduce_probe<<<1, 1>>>(
        reinterpret_cast<const std::uint32_t*>(multicast_pointer),
        outputs[index]);
    RUNTIME_CHECK("multimem.warmup.launch", cudaGetLastError());
    RUNTIME_CHECK("multimem.warmup.synchronize", cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    RUNTIME_CHECK("cudaEventCreate.start", cudaEventCreate(&start));
    RUNTIME_CHECK("cudaEventCreate.stop", cudaEventCreate(&stop));
    RUNTIME_CHECK("cudaEventRecord.start", cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      multimem_load_reduce_probe<<<1, 1>>>(
          reinterpret_cast<const std::uint32_t*>(multicast_pointer),
          outputs[index]);
    }
    RUNTIME_CHECK("multimem.timed.launch", cudaGetLastError());
    RUNTIME_CHECK("cudaEventRecord.stop", cudaEventRecord(stop));
    RUNTIME_CHECK("cudaEventSynchronize.stop", cudaEventSynchronize(stop));
    float total_ms = 0.0F;
    RUNTIME_CHECK("cudaEventElapsedTime",
                  cudaEventElapsedTime(&total_ms, start, stop));
    latency_us[index] = total_ms * 1000.0F / iterations;
    RUNTIME_CHECK("cudaMemcpy.output",
                  cudaMemcpy(&observed[index], outputs[index],
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    RUNTIME_CHECK("cudaEventDestroy.start", cudaEventDestroy(start));
    RUNTIME_CHECK("cudaEventDestroy.stop", cudaEventDestroy(stop));
  }

  std::uint32_t backing_values[device_count]{};
  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.read_backing", cudaSetDevice(index));
    RUNTIME_CHECK(
        "cudaMemcpy.backing",
        cudaMemcpy(&backing_values[index],
                   reinterpret_cast<void*>(unicast_pointers[index]),
                   sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  }

  const std::uint32_t expected = first + second;
  const bool passed = observed[0] == expected && observed[1] == expected &&
                      backing_values[0] == first &&
                      backing_values[1] == second;
  std::printf(
      "{\"status\":\"%s\",\"visible_device_count\":%d,"
      "\"multicast_attributes\":[%d,%d],"
      "\"input_values\":[%u,%u],\"backing_values\":[%u,%u],"
      "\"expected_reduce\":%u,\"observed_reduce\":[%u,%u],"
      "\"latency_us\":[%.6f,%.6f],\"iterations\":%d,"
      "\"allocation_size_bytes\":%zu,"
      "\"multicast_minimum_granularity_bytes\":%zu,"
      "\"multicast_recommended_granularity_bytes\":%zu,"
      "\"allocation_granularity_bytes\":[%zu,%zu]}\n",
      passed ? "passed" : "failed", visible_devices,
      multicast_attributes[0], multicast_attributes[1], first, second,
      backing_values[0], backing_values[1], expected, observed[0], observed[1],
      latency_us[0], latency_us[1], iterations, allocation_size,
      multicast_minimum, multicast_recommended, allocation_granularities[0],
      allocation_granularities[1]);

  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.cleanup_output", cudaSetDevice(index));
    RUNTIME_CHECK("cudaFree.output", cudaFree(outputs[index]));
  }
  RUNTIME_CHECK("cudaSetDevice.cleanup_multicast", cudaSetDevice(0));
  DRIVER_CHECK("cuMemUnmap.multicast",
               cuMemUnmap(multicast_pointer, allocation_size));
  DRIVER_CHECK("cuMemAddressFree.multicast",
               cuMemAddressFree(multicast_pointer, allocation_size));
  for (int index = 0; index < device_count; ++index) {
    RUNTIME_CHECK("cudaSetDevice.cleanup_unicast", cudaSetDevice(index));
    DRIVER_CHECK("cuMulticastUnbind",
                 cuMulticastUnbind(multicast_handle, devices[index], 0,
                                   allocation_size));
    DRIVER_CHECK("cuMemUnmap.unicast",
                 cuMemUnmap(unicast_pointers[index], allocation_size));
    DRIVER_CHECK("cuMemRelease.unicast",
                 cuMemRelease(unicast_handles[index]));
    DRIVER_CHECK("cuMemAddressFree.unicast",
                 cuMemAddressFree(unicast_pointers[index], allocation_size));
  }
  DRIVER_CHECK("cuMemRelease.multicast", cuMemRelease(multicast_handle));
  return passed ? 0 : 1;
}

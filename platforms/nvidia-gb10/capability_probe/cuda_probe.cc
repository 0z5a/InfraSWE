#include <cuda.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main() {
  CUresult init = cuInit(0);
  CUdevice device = 0;
  CUresult selected = init == CUDA_SUCCESS ? cuDeviceGet(&device, 0) : init;
  std::cout << "{\"available\":" << (selected == CUDA_SUCCESS ? "true" : "false")
            << ",\"cu_init_returncode\":" << static_cast<int>(init)
            << ",\"cu_device_get_returncode\":" << static_cast<int>(selected)
            << ",\"attributes\":{";
  const std::vector<std::pair<std::string, CUdevice_attribute>> attributes = {
      {"unified_addressing", CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
      {"concurrent_managed_access", CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS},
      {"pageable_memory_access", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
      {"host_page_table_coherence",
       CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES},
      {"direct_managed_mem_access_from_host",
       CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST},
      {"host_native_atomics", CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
      {"cluster_launch", CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH},
      {"gpudirect_rdma_supported", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
      {"dma_buf_supported", CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED},
      {"async_engine_count", CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
      {"tensor_map_access_supported", CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED},
  };
  for (size_t index = 0; index < attributes.size(); ++index) {
    int value = 0;
    CUresult result = selected == CUDA_SUCCESS
                          ? cuDeviceGetAttribute(&value, attributes[index].second, device)
                          : selected;
    if (index != 0) std::cout << ',';
    std::cout << '\"' << attributes[index].first << "\":{\"returncode\":"
              << static_cast<int>(result) << ",\"value\":";
    if (result == CUDA_SUCCESS)
      std::cout << value;
    else
      std::cout << "null";
    std::cout << '}';
  }
  std::cout << "}}\n";
  return selected == CUDA_SUCCESS ? 0 : 2;
}

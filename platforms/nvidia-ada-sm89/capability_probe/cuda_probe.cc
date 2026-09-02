#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string escape_json(const char* value) {
  std::string escaped;
  for (const char character : std::string(value)) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

}  // namespace

int main(int argc, char** argv) {
  const int device_index = argc > 1 ? std::atoi(argv[1]) : 0;
  cudaError_t selected = cudaSetDevice(device_index);
  cudaDeviceProp properties{};
  cudaError_t properties_result =
      selected == cudaSuccess ? cudaGetDeviceProperties(&properties, device_index) : selected;

  std::cout << "{\"available\":" << (properties_result == cudaSuccess ? "true" : "false")
            << ",\"device_index\":" << device_index
            << ",\"cuda_returncode\":" << static_cast<int>(properties_result);
  if (properties_result == cudaSuccess) {
    std::cout << ",\"name\":\"" << escape_json(properties.name) << "\""
              << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
              << "\""
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"warp_size\":" << properties.warpSize
              << ",\"l2_bytes\":" << properties.l2CacheSize
              << ",\"framebuffer_bytes\":" << properties.totalGlobalMem
              << ",\"shared_mem_per_sm_bytes\":" << properties.sharedMemPerMultiprocessor
              << ",\"max_shared_mem_per_block_optin_bytes\":"
              << properties.sharedMemPerBlockOptin
              << ",\"registers_per_sm_32bit\":" << properties.regsPerMultiprocessor
              << ",\"max_registers_per_block_32bit\":" << properties.regsPerBlock
              << ",\"max_threads_per_sm\":" << properties.maxThreadsPerMultiProcessor
              << ",\"async_engine_count\":" << properties.asyncEngineCount
              << ",\"unified_addressing\":" << properties.unifiedAddressing
              << ",\"managed_memory\":" << properties.managedMemory
              << ",\"concurrent_managed_access\":" << properties.concurrentManagedAccess
              << ",\"pageable_memory_access\":" << properties.pageableMemoryAccess
              << ",\"host_page_table_coherence\":"
              << properties.pageableMemoryAccessUsesHostPageTables
              << ",\"host_native_atomics\":" << properties.hostNativeAtomicSupported
              << ",\"persisting_l2_cache_max_bytes\":"
              << properties.persistingL2CacheMaxSize
              << ",\"access_policy_max_window_bytes\":"
              << properties.accessPolicyMaxWindowSize
              << ",\"pci_domain_id\":" << properties.pciDomainID
              << ",\"pci_bus_id\":" << properties.pciBusID
              << ",\"pci_device_id\":" << properties.pciDeviceID;
  }

  const std::vector<std::pair<std::string, cudaDeviceAttr>> attributes = {
      {"ecc_enabled", cudaDevAttrEccEnabled},
      {"memory_clock_khz", cudaDevAttrMemoryClockRate},
      {"max_blocks_per_sm", cudaDevAttrMaxBlocksPerMultiprocessor},
      {"max_shared_mem_per_block_optin_bytes", cudaDevAttrMaxSharedMemoryPerBlockOptin},
      {"memory_pools_supported", cudaDevAttrMemoryPoolsSupported},
      {"cooperative_launch", cudaDevAttrCooperativeLaunch},
      {"compute_preemption_supported", cudaDevAttrComputePreemptionSupported},
  };
  std::cout << ",\"attributes\":{";
  for (size_t index = 0; index < attributes.size(); ++index) {
    int value = 0;
    cudaError_t result = properties_result == cudaSuccess
                             ? cudaDeviceGetAttribute(&value, attributes[index].second, device_index)
                             : properties_result;
    if (index != 0) std::cout << ',';
    std::cout << '"' << attributes[index].first << "\":{\"returncode\":"
              << static_cast<int>(result) << ",\"value\":";
    if (result == cudaSuccess)
      std::cout << value;
    else
      std::cout << "null";
    std::cout << '}';
  }
  std::cout << "}}\n";
  return properties_result == cudaSuccess ? 0 : 2;
}

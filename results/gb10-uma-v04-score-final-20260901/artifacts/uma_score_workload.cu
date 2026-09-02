#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string mode{"candidate"};
  std::string allocation{"system"};
  std::string touch{"cpu-first"};
  std::string protocol_id{"gb10-uma-load-normalized-v0.4-r1"};
  std::string regime{"normal"};
  std::string samples_path;
  int replay_index{1};
  int requests{1200};
  int elements{65536};
  int streams{4};
  int tenants{4};
  double arrival_rate{0.0};
  double slo_us{5000.0};
  bool burst{false};
  std::uint64_t max_workspace_bytes{1ULL << 30};
  int required_runtime_version{13000};
};

struct RequestRecord {
  int request_id{0};
  int tenant_id{0};
  double offered_at{0.0};
  double completed_at{0.0};
  double latency{0.0};
  bool completed{false};
  bool output_valid{false};
  bool slo_met{false};
  std::string error_code;
};

double seconds_since(Clock::time_point start, Clock::time_point point) {
  return std::chrono::duration<double>(point - start).count();
}

std::string argument_value(int& index, int argc, char** argv) {
  if (index + 1 >= argc) throw std::runtime_error("missing argument value");
  return argv[++index];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--mode") options.mode = argument_value(index, argc, argv);
    else if (key == "--allocation") options.allocation = argument_value(index, argc, argv);
    else if (key == "--touch") options.touch = argument_value(index, argc, argv);
    else if (key == "--protocol-id") options.protocol_id = argument_value(index, argc, argv);
    else if (key == "--regime") options.regime = argument_value(index, argc, argv);
    else if (key == "--samples") options.samples_path = argument_value(index, argc, argv);
    else if (key == "--replay-index")
      options.replay_index = std::stoi(argument_value(index, argc, argv));
    else if (key == "--requests")
      options.requests = std::stoi(argument_value(index, argc, argv));
    else if (key == "--elements")
      options.elements = std::stoi(argument_value(index, argc, argv));
    else if (key == "--streams")
      options.streams = std::stoi(argument_value(index, argc, argv));
    else if (key == "--tenants")
      options.tenants = std::stoi(argument_value(index, argc, argv));
    else if (key == "--arrival-rate")
      options.arrival_rate = std::stod(argument_value(index, argc, argv));
    else if (key == "--slo-us")
      options.slo_us = std::stod(argument_value(index, argc, argv));
    else if (key == "--max-workspace-bytes")
      options.max_workspace_bytes = std::stoull(argument_value(index, argc, argv));
    else if (key == "--required-runtime-version")
      options.required_runtime_version = std::stoi(argument_value(index, argc, argv));
    else if (key == "--burst")
      options.burst = true;
    else
      throw std::runtime_error("unknown argument: " + key);
  }
  if (options.mode != "candidate" && options.mode != "reference")
    throw std::runtime_error("mode must be candidate or reference");
  if (options.allocation != "system" && options.allocation != "managed" &&
      options.allocation != "pinned")
    throw std::runtime_error("allocation must be system, managed, or pinned");
  if (options.touch != "cpu-first" && options.touch != "gpu-first")
    throw std::runtime_error("touch must be cpu-first or gpu-first");
  if (options.requests < 1 || options.elements < 1 || options.streams < 1 ||
      options.tenants < 1 || options.arrival_rate < 0 || options.slo_us <= 0)
    throw std::runtime_error("numeric arguments are outside their valid range");
  return options;
}

void print_structured_failure(const std::string& code, const std::string& reason) {
  std::cout << "{\"schema_version\":\"0.4\",\"passed\":false,\"failure_code\":\""
            << code << "\",\"reason\":\"" << reason << "\"}\n";
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::size_t resident_bytes() {
  std::ifstream stream("/proc/self/statm");
  std::size_t total_pages = 0;
  std::size_t resident_pages = 0;
  stream >> total_pages >> resident_pages;
  (void)total_pages;
  return resident_pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
}

float input_value(std::size_t index) {
  return static_cast<float>(index % 251) * 0.001f;
}

__global__ void initialize_kernel(float* values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] = static_cast<float>(index % 251) * 0.001f;
}

extern "C" __global__ void uma_transform_kernel(const float* input, float* output,
                                                   int count, std::uint64_t request_id) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float request_term = static_cast<float>(request_id % 17) * 0.0001f;
    output[index] = fmaf(input[index], 1.25f, request_term);
  }
}

struct Slot {
  float* host_input{nullptr};
  float* host_output{nullptr};
  float* device_input{nullptr};
  float* device_output{nullptr};
  cudaStream_t stream{nullptr};
  std::string mode;
  std::string allocation;
  std::size_t bytes{0};

  Slot(const Options& options, std::size_t allocation_bytes)
      : mode(options.mode), allocation(options.allocation), bytes(allocation_bytes) {
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    if (mode == "reference") {
      host_input = static_cast<float*>(std::aligned_alloc(4096, bytes));
      host_output = static_cast<float*>(std::aligned_alloc(4096, bytes));
      if (!host_input || !host_output) throw std::bad_alloc();
      check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc input");
      check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc output");
    } else if (allocation == "system") {
      host_input = static_cast<float*>(std::aligned_alloc(4096, bytes));
      host_output = static_cast<float*>(std::aligned_alloc(4096, bytes));
      if (!host_input || !host_output) throw std::bad_alloc();
      device_input = host_input;
      device_output = host_output;
    } else if (allocation == "managed") {
      check_cuda(cudaMallocManaged(&host_input, bytes), "cudaMallocManaged input");
      check_cuda(cudaMallocManaged(&host_output, bytes), "cudaMallocManaged output");
      device_input = host_input;
      device_output = host_output;
    } else {
      check_cuda(
          cudaHostAlloc(&host_input, bytes, cudaHostAllocMapped), "cudaHostAlloc input");
      check_cuda(
          cudaHostAlloc(&host_output, bytes, cudaHostAllocMapped), "cudaHostAlloc output");
      check_cuda(
          cudaHostGetDevicePointer(&device_input, host_input, 0), "cudaHostGetDevicePointer input");
      check_cuda(cudaHostGetDevicePointer(&device_output, host_output, 0),
                 "cudaHostGetDevicePointer output");
    }
    if (options.touch == "gpu-first" && mode != "reference") {
      initialize_kernel<<<(options.elements + 255) / 256, 256, 0, stream>>>(device_input,
                                                                            options.elements);
      check_cuda(cudaStreamSynchronize(stream), "gpu-first initialization");
    } else {
      for (int index = 0; index < options.elements; ++index) {
        host_input[index] = input_value(static_cast<std::size_t>(index));
      }
    }
    std::memset(host_output, 0, bytes);
  }

  Slot(const Slot&) = delete;
  Slot& operator=(const Slot&) = delete;

  ~Slot() {
    if (stream) cudaStreamDestroy(stream);
    if (mode == "reference") {
      if (device_input) cudaFree(device_input);
      if (device_output) cudaFree(device_output);
      std::free(host_input);
      std::free(host_output);
    } else if (allocation == "system") {
      std::free(host_input);
      std::free(host_output);
    } else if (allocation == "managed") {
      if (host_input) cudaFree(host_input);
      if (host_output) cudaFree(host_output);
    } else {
      if (host_input) cudaFreeHost(host_input);
      if (host_output) cudaFreeHost(host_output);
    }
  }
};

bool validate_output(const Slot& slot, int elements, int request_id) {
  const int probes[] = {0, elements / 2, elements - 1};
  const float request_term = static_cast<float>(request_id % 17) * 0.0001f;
  for (const int index : probes) {
    const float expected = std::fma(input_value(static_cast<std::size_t>(index)), 1.25f,
                                    request_term);
    if (std::abs(slot.host_output[index] - expected) > 1e-6f) return false;
  }
  return true;
}

void execute_request(Slot& slot, const Options& options, int request_id) {
  if (options.mode == "reference") {
    check_cuda(cudaMemcpyAsync(slot.device_input, slot.host_input, slot.bytes,
                               cudaMemcpyHostToDevice, slot.stream),
               "reference H2D");
  }
  uma_transform_kernel<<<(options.elements + 255) / 256, 256, 0, slot.stream>>>(
      slot.device_input, slot.device_output, options.elements,
      static_cast<std::uint64_t>(request_id));
  check_cuda(cudaGetLastError(), "uma_transform_kernel launch");
  if (options.mode == "reference") {
    check_cuda(cudaMemcpyAsync(slot.host_output, slot.device_output, slot.bytes,
                               cudaMemcpyDeviceToHost, slot.stream),
               "reference D2H");
  }
  check_cuda(cudaStreamSynchronize(slot.stream), "request synchronization");
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double jain_index(const std::vector<double>& values) {
  double sum = 0.0;
  double squares = 0.0;
  for (const double value : values) {
    sum += value;
    squares += value * value;
  }
  if (values.empty() || squares == 0.0) return 0.0;
  return (sum * sum) / (static_cast<double>(values.size()) * squares);
}

void write_samples(const Options& options, const std::vector<RequestRecord>& records) {
  if (options.samples_path.empty()) return;
  std::ofstream stream(options.samples_path);
  if (!stream) throw std::runtime_error("cannot open request sample output");
  stream << std::setprecision(12);
  for (const auto& record : records) {
    stream << "{\"schema_version\":\"0.4\",\"protocol_id\":\""
           << options.protocol_id << "\",\"replay_index\":" << options.replay_index
           << ",\"regime\":\"" << options.regime << "\",\"request_id\":\"r"
           << options.replay_index << "-" << record.request_id << "\",\"tenant_id\":\"t"
           << record.tenant_id << "\",\"offered_at_seconds\":" << record.offered_at
           << ",\"completed_at_seconds\":" << record.completed_at
           << ",\"latency_seconds\":" << record.latency
           << ",\"completed\":" << (record.completed ? "true" : "false")
           << ",\"output_valid\":" << (record.output_valid ? "true" : "false")
           << ",\"slo_met\":" << (record.slo_met ? "true" : "false")
           << ",\"error_code\":";
    if (record.error_code.empty())
      stream << "null";
    else
      stream << "\"" << record.error_code << "\"";
    stream << "}\n";
  }
}

int run(const Options& options) {
  int runtime_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  if (runtime_version < options.required_runtime_version) {
    print_structured_failure("GB10_RUNTIME_VERSION_UNSUPPORTED", "runtime below frozen minimum");
    return 3;
  }
  int pageable = 0;
  int host_tables = 0;
  int unified = 0;
  check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0),
             "pageable memory attribute");
  check_cuda(cudaDeviceGetAttribute(&host_tables,
                                    cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0),
             "host page table attribute");
  check_cuda(cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, 0),
             "unified addressing attribute");
  if (options.mode == "candidate" && options.allocation == "system" &&
      (std::getenv("INFRASWE_FORCE_NO_PAGEABLE") || !pageable || !host_tables || !unified)) {
    print_structured_failure("GB10_PAGEABLE_ACCESS_UNAVAILABLE",
                             "system-memory GPU path rejected by capability contract");
    return 4;
  }
  if (std::getenv("INFRASWE_FORCE_ALLOC_FAIL")) {
    print_structured_failure("GB10_ALLOCATION_INJECTED_FAILURE",
                             "allocator failure injection was handled before launch");
    return 5;
  }

  const std::size_t raw_bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
  const std::size_t bytes = ((raw_bytes + 4095) / 4096) * 4096;
  const std::uint64_t per_slot = options.mode == "reference" ? 4ULL * bytes : 2ULL * bytes;
  const std::uint64_t required_workspace = per_slot * static_cast<std::uint64_t>(options.streams);
  if (required_workspace > options.max_workspace_bytes) {
    print_structured_failure("GB10_WORKSPACE_BUDGET_EXCEEDED",
                             "required workspace exceeds frozen task budget");
    return 6;
  }

  std::vector<std::unique_ptr<Slot>> slots;
  slots.reserve(static_cast<std::size_t>(options.streams));
  for (int index = 0; index < options.streams; ++index) {
    slots.push_back(std::make_unique<Slot>(options, bytes));
    const int warmup_request_id = 100000 + index;
    execute_request(*slots.back(), options, warmup_request_id);
    if (!validate_output(*slots.back(), options.elements, warmup_request_id)) {
      print_structured_failure("GB10_WARMUP_CORRECTNESS_FAILED", "warmup output mismatch");
      return 7;
    }
  }

  const std::size_t rss_start = resident_bytes();
  std::vector<RequestRecord> records(static_cast<std::size_t>(options.requests));
  std::deque<int> queue;
  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  bool producer_done = false;
  std::size_t max_queue_depth = 0;
  std::atomic<int> worker_failures{0};
  const auto start = Clock::now();

  std::vector<std::thread> workers;
  for (int worker_index = 0; worker_index < options.streams; ++worker_index) {
    workers.emplace_back([&, worker_index] {
      Slot& slot = *slots[static_cast<std::size_t>(worker_index)];
      while (true) {
        int request_id = -1;
        {
          std::unique_lock lock(queue_mutex);
          queue_ready.wait(lock, [&] { return producer_done || !queue.empty(); });
          if (queue.empty() && producer_done) break;
          request_id = queue.front();
          queue.pop_front();
        }
        auto& record = records[static_cast<std::size_t>(request_id)];
        try {
          execute_request(slot, options, request_id);
          record.output_valid = validate_output(slot, options.elements, request_id);
          if (!record.output_valid) record.error_code = "GB10_OUTPUT_MISMATCH";
        } catch (const std::exception&) {
          record.error_code = "GB10_CUDA_REQUEST_FAILED";
          ++worker_failures;
        }
        const auto completed = Clock::now();
        record.completed = true;
        record.completed_at = seconds_since(start, completed);
        record.latency = record.completed_at - record.offered_at;
        record.slo_met = record.output_valid && record.latency <= options.slo_us * 1e-6;
      }
    });
  }

  for (int request_id = 0; request_id < options.requests; ++request_id) {
    if (options.arrival_rate > 0.0) {
      double target_seconds = static_cast<double>(request_id) / options.arrival_rate;
      if (options.burst) {
        constexpr int burst_size = 20;
        const int group = request_id / burst_size;
        const int within = request_id % burst_size;
        target_seconds = static_cast<double>(group * burst_size) / options.arrival_rate +
                         static_cast<double>(within) / (10.0 * options.arrival_rate);
      }
      std::this_thread::sleep_until(start + std::chrono::duration_cast<Clock::duration>(
                                               std::chrono::duration<double>(target_seconds)));
    }
    auto& record = records[static_cast<std::size_t>(request_id)];
    record.request_id = request_id;
    record.tenant_id = request_id % options.tenants;
    record.offered_at = seconds_since(start, Clock::now());
    {
      std::lock_guard lock(queue_mutex);
      queue.push_back(request_id);
      max_queue_depth = std::max(max_queue_depth, queue.size());
    }
    queue_ready.notify_one();
  }

  std::size_t queue_at_offer_end = 0;
  const double offer_end_seconds = seconds_since(start, Clock::now());
  {
    std::lock_guard lock(queue_mutex);
    queue_at_offer_end = queue.size();
    producer_done = true;
  }
  queue_ready.notify_all();
  for (auto& worker : workers) worker.join();
  const double completed_seconds = seconds_since(start, Clock::now());
  const std::size_t rss_end = resident_bytes();
  const std::size_t rss_growth = rss_end > rss_start ? rss_end - rss_start : 0;

  int completed = 0;
  int valid = 0;
  int slo_good = 0;
  std::vector<double> latencies;
  std::vector<double> tenant_goodput(static_cast<std::size_t>(options.tenants), 0.0);
  for (const auto& record : records) {
    if (record.completed) ++completed;
    if (record.output_valid) ++valid;
    if (record.slo_met) {
      ++slo_good;
      tenant_goodput[static_cast<std::size_t>(record.tenant_id)] += 1.0;
    }
    if (record.completed) latencies.push_back(record.latency);
  }
  write_samples(options, records);
  const bool passed = completed == options.requests && valid == options.requests &&
                      worker_failures.load() == 0;
  const double throughput = completed_seconds > 0.0 ? completed / completed_seconds : 0.0;
  std::cout << std::setprecision(12)
            << "{\"schema_version\":\"0.4\",\"passed\":"
            << (passed ? "true" : "false") << ",\"mode\":\"" << options.mode
            << "\",\"allocation\":\"" << options.allocation << "\",\"touch\":\""
            << options.touch << "\",\"requests\":" << options.requests
            << ",\"completed_requests\":" << completed << ",\"valid_requests\":" << valid
            << ",\"slo_good_requests\":" << slo_good << ",\"slo_goodput_ratio\":"
            << static_cast<double>(slo_good) / options.requests
            << ",\"error_drop_rate\":"
            << static_cast<double>(options.requests - valid) / options.requests
            << ",\"throughput_rps\":" << throughput << ",\"p50_seconds\":"
            << percentile(latencies, 0.50) << ",\"p95_seconds\":"
            << percentile(latencies, 0.95) << ",\"p99_seconds\":"
            << percentile(latencies, 0.99) << ",\"duration_seconds\":" << completed_seconds
            << ",\"offer_duration_seconds\":" << offer_end_seconds
            << ",\"drain_seconds\":" << std::max(0.0, completed_seconds - offer_end_seconds)
            << ",\"max_queue_depth\":" << max_queue_depth
            << ",\"queue_depth_at_offer_end\":" << queue_at_offer_end
            << ",\"rss_growth_bytes\":" << rss_growth << ",\"fairness_jain\":"
            << jain_index(tenant_goodput) << ",\"workspace_bytes\":" << required_workspace
            << ",\"minimum_external_bytes_per_request\":" << 2ULL * raw_bytes
            << ",\"runtime_version\":" << runtime_version
            << ",\"pageable_memory_access\":" << pageable
            << ",\"host_page_table_coherence\":" << host_tables
            << ",\"unified_addressing\":" << unified
            << ",\"cpu_dereferenced_cuda_malloc\":false}\n";
  return passed ? 0 : 8;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::bad_alloc&) {
    print_structured_failure("GB10_ALLOCATION_FAILED", "host allocation failed");
    return 9;
  } catch (const std::exception& error) {
    print_structured_failure("GB10_WORKLOAD_ERROR", error.what());
    return 10;
  }
}

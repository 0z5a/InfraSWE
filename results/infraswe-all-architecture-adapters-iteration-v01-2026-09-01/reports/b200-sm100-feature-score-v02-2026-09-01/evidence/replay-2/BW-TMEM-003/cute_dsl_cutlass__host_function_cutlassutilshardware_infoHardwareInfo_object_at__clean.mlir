module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass__empty_kernel_cutlassutilshardware_infoHardwareInfo_object_at__0() attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 1, 1, 1>} {
      return
    }
  }
  func.func @cutlass__host_function_cutlassutilshardware_infoHardwareInfo_object_at_() -> i32 attributes {llvm.emit_c_interface} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = cuda.cast %c0_i64 : i64 -> !cuda.stream
    %1 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c1_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %c0_i64, gridDim = (%c1_i32, %c1_i32, %c1_i32), stream = %0) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%1] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cooperative[%1] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %2 = cuda.launch_ex @kernels::@kernel_cutlass__empty_kernel_cutlassutilshardware_infoHardwareInfo_object_at__0<%1> () {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, () -> !cuda.result
    %3 = cuda.cast %2 : !cuda.result -> i32
    cuda.return_if_error %3 : i32
    return %c0_i32 : i32
  }
}


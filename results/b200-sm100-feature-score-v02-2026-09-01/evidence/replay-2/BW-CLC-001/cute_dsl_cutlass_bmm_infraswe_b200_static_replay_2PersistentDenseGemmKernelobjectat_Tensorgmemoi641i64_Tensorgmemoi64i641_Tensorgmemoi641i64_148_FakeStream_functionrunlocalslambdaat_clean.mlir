!copy_ldtm_256 = !cute.tiled_copy<!cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x4>, layout_copy_tv = <"((32,4),(32,16)):((0,1),(64,4))">, tiler_mn = <"[(4,16):(32,1);32:1]">>
!copy_stsm_4 = !cute.tiled_copy<!cute_nvgpu.atom.stsm<f16, mode = <"(8,8)">, num_matrices = 4, t>, layout_copy_tv = <"((4,8,4),((2,2,4),1)):((128,4,1),((64,32,512),0))">, tiler_mn = <"[(4,16):(32,1);32:1]">>
!memref_gmem_f16 = !cute.memref<f16, gmem, align<16>, "(?,?,?):(?{i64},1,?{i64})">
!memref_gmem_f16_1 = !cute.memref<f16, gmem, align<16>, "(?,?,?):(?{i64},?{i64},1)">
!memref_gmem_f16_2 = !cute.memref<f16, gmem, align<16>, "(?,?,?):(1,?{i64},?{i64})">
!memref_gmem_f16_3 = !cute.memref<f16, gmem, align<16>, "((128,128),(?,?,?)):((1,?{i64}),(128,?{i64 div=128},?{i64}))">
!memref_gmem_f16_4 = !cute.memref<f16, gmem, align<16>, "(?,?,?):(128,?{i64 div=128},?{i64})">
!memref_rmem_f16 = !cute.memref<f16, rmem, align<32>, "(((2,2,4),1),2,1):(((1,2,4),0),16,0)">
!memref_rmem_f16_1 = !cute.memref<f16, rmem, align<32>, "((8,2),2,1):((1,8),16,0)">
!memref_rmem_f16_2 = !cute.memref<f16, rmem, align<32>, "((8,2),(2,1)):((1,8),(16,0))">
!memref_rmem_f32 = !cute.memref<f32, rmem, align<32>, "(((2,2,4),1),2,1):(((1,2,4),0),16,0)">
!memref_rmem_f32_1 = !cute.memref<f32, rmem, align<32>, "(((2,2,4),1),(2,1)):(((1,2,4),0),(16,0))">
!memref_rmem_f32_2 = !cute.memref<f32, rmem, align<32>, "((8,2),2,1):((1,8),16,0)">
!memref_smem_f16 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "(((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
!memref_smem_f16_1 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((((64,2),16),1,4),6):((((1,4096),64),0,1024),8192)">
!memref_smem_f16_2 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((4096,2),6):((1,4096),8192)">
!memref_smem_f16_3 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((4096,2)):((1,4096))">
!memref_smem_f16_4 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((4096,2),1):((1,4096),0)">
!memref_smem_f16_5 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((4096,2),(1)):((1,4096),(0))">
!memref_smem_f16_6 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">
!memref_smem_f16_7 = !cute.memref<f16, smem, align<16>, S<3,4,3>, "((8,2),2,1,(1,4)):((1,1024),16,0,(0,4096))">
!memref_smem_f16_8 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "(((64,2),(8,4)),(1,4)):(((1,2048),(64,512)),(0,4096))">
!memref_smem_f16_9 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((2048,2),(1,4)):((1,2048),(0,4096))">
!memref_smem_f16_10 = !cute.memref<f16, smem, align<16>, S<3,4,3>, "((8,2),2,1):((1,1024),16,0)">
!memref_smem_f16_11 = !cute.memref<f16, smem, align<16>, S<3,4,3>, "((8,2),(2,1)):((1,1024),(16,0))">
!memref_smem_f16_12 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((2048,2)):((1,2048))">
!memref_smem_f16_13 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((2048,2),1):((1,2048),0)">
!memref_smem_f16_14 = !cute.memref<f16, smem, align<128>, S<3,4,3>, "((2048,2),(1)):((1,2048),(0))">
!memref_tmem_f32 = !cute.memref<f32, tmem, align<1>, "((128,128),1,1,2):((65536,1),0,0,128)">
!memref_tmem_f32_1 = !cute.memref<f32, tmem, align<16>, "((128,128),1,1,2):((65536,1),0,0,128)">
!memref_tmem_f32_2 = !cute.memref<f32, tmem, align<16>, "((128,128),1,1):((65536,1),0,0)">
!memref_tmem_f32_3 = !cute.memref<f32, tmem, align<16>, "((128,1),(128,1),2):((65536,0),(1,0),128)">
!memref_tmem_f32_4 = !cute.memref<f32, tmem, align<16>, "(128,32,1,4,2):(65536,1,0,32,128)">
!memref_tmem_f32_5 = !cute.memref<f32, tmem, align<16>, "(128,32):(65536,1)">
!memref_tmem_f32_6 = !cute.memref<f32, tmem, align<16>, "(((32,16),1),2,1,1,4,2):(((1,65536),0),1048576,0,0,32,128)">
!memref_tmem_f32_7 = !cute.memref<f32, tmem, align<16>, "(((32,16),1),2,1,1,4):(((1,65536),0),1048576,0,0,32)">
!memref_tmem_f32_8 = !cute.memref<f32, tmem, align<16>, "(((32,16),1),2,1,(1,4)):(((1,65536),0),1048576,0,(0,32))">
!memref_tmem_f32_9 = !cute.memref<f32, tmem, align<16>, "(((32,16),1),2,1):(((1,65536),0),1048576,0)">
!memref_tmem_f32_10 = !cute.memref<f32, tmem, align<16>, "(((32,16),1),(2,1)):(((1,65536),0),(1048576,0))">
!mma_f16_f16_f32_128x128x16 = !cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (mn, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0>, atom_layout_MNK = <"(1,1,1):(0,0,0)">>
#loop_unroll = #llvm.loop_unroll<disable = true, count = 1 : i32>
#loop_unroll1 = #llvm.loop_unroll<full = true>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>
#loop_annotation1 = #llvm.loop_annotation<unroll = #loop_unroll1>
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_kernel_infraswe_b200_static_replay_2PersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA11281612_0(%arg0: !mma_f16_f16_f32_128x128x16, %arg1: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg2: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg3: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg4: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg5: !cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg6: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg7: !cute.layout<"((1),1,1,1):((0),0,0,0)">, %arg8: !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, %arg9: !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, %arg10: !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">, %arg11: !cute.layout<"128:1">, %arg12: !cute.layout<"32:1">, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: !cute.fast_divmod_divisor<32>, %arg17: !cute.fast_divmod_divisor<32>) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 192, 1, 1>} {
      %c127_i32 = arith.constant 127 : i32
      %c256_i32 = arith.constant 256 : i32
      %c229632_i32 = arith.constant 229632 : i32
      %false = arith.constant false
      %c160_i32 = arith.constant 160 : i32
      %c2_i32 = arith.constant 2 : i32
      %c6_i32 = arith.constant 6 : i32
      %c32768_i32 = arith.constant 32768 : i32
      %true = arith.constant true
      %c10000000_i32 = arith.constant 10000000 : i32
      %c196864_i32 = arith.constant 196864 : i32
      %c98560_i32 = arith.constant 98560 : i32
      %c-128_i32 = arith.constant -128 : i32
      %c128_i32 = arith.constant 128 : i32
      %c4_i32 = arith.constant 4 : i32
      %c144_i32 = arith.constant 144 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c5_i32 = arith.constant 5 : i32
      %c32_i32 = arith.constant 32 : i32
      %int_tuple = cute.make_int_tuple(%arg13, %arg14, %arg15) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
      %tile = cute.make_tile() : () -> !cute.tile<"[1:0;1:0]">
      %shp = cute.ceil_div(%int_tuple, %tile) : !cute.int_tuple<"(?,?,?)">, !cute.tile<"[1:0;1:0]">
      %e0, %e1, %e2 = cute.get_leaves(%shp) : !cute.int_tuple<"(?,?,?)">
      %shape = cute.make_shape(%e0, %e1, %e2) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
      %lay = cute.make_layout(%shape) : !cute.layout<"(?,?,?):(1,?,?)">
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.tid.y : i32
      %2 = nvvm.read.ptx.sreg.tid.z : i32
      %3 = nvvm.read.ptx.sreg.ntid.x : i32
      %4 = nvvm.read.ptx.sreg.ntid.y : i32
      %5 = arith.muli %1, %3 : i32
      %6 = arith.addi %0, %5 : i32
      %7 = arith.muli %2, %3 : i32
      %8 = arith.muli %7, %4 : i32
      %9 = arith.addi %6, %8 : i32
      %10 = arith.floordivsi %9, %c32_i32 : i32
      %11 = cute_nvgpu.arch.make_warp_uniform(%10) : i32
      %12 = arith.cmpi eq, %11, %c5_i32 : i32
      scf.if %12 {
        cute_nvgpu.prefetch_tma_desc(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> ()
        cute_nvgpu.prefetch_tma_desc(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> ()
        cute_nvgpu.prefetch_tma_desc(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> ()
      }
      %smem_ptr = cute_nvgpu.arch.get_dyn_smem() : !cute.ptr<i8, smem, align<1024>>
      %int_tuple_0 = cute.make_int_tuple() : () -> !cute.int_tuple<"144">
      %ptr = cute.add_offset(%smem_ptr, %int_tuple_0) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"144">) -> !cute.ptr<i8, smem, align<16>>
      %smem_size = cute_nvgpu.arch.get_dyn_smem_size() : i32
      %13 = arith.cmpi sge, %smem_size, %c144_i32 : i32
      cf.assert %13, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 144 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %int_tuple_1 = cute.make_int_tuple() : () -> !cute.int_tuple<"96">
      %ptr_2 = cute.add_offset(%smem_ptr, %int_tuple_1) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"96">) -> !cute.ptr<i8, smem, align<32>>
      %int_tuple_3 = cute.make_int_tuple() : () -> !cute.int_tuple<"136">
      %ptr_4 = cute.add_offset(%smem_ptr, %int_tuple_3) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"136">) -> !cute.ptr<i8, smem, align<8>>
      %iter = cute.recast_iter(%ptr_4) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i32, smem, align<8>>
      %iter_5 = cute.recast_iter(%smem_ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<i64, smem, align<1024>>
      %14 = arith.cmpi eq, %11, %c0_i32 : i32
      scf.if %14 {
        %41 = builtin.unrealized_conversion_cast %iter_5 : !cute.ptr<i64, smem, align<1024>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %41, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_50 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
        %ptr_51 = cute.add_offset(%iter_5, %int_tuple_50) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
        %42 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_52 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
        %ptr_53 = cute.add_offset(%iter_5, %int_tuple_52) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem, align<16>>
        %43 = builtin.unrealized_conversion_cast %ptr_53 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_54 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
        %ptr_55 = cute.add_offset(%iter_5, %int_tuple_54) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
        %44 = builtin.unrealized_conversion_cast %ptr_55 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %44, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_56 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
        %ptr_57 = cute.add_offset(%iter_5, %int_tuple_56) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"4">) -> !cute.ptr<i64, smem, align<32>>
        %45 = builtin.unrealized_conversion_cast %ptr_57 : !cute.ptr<i64, smem, align<32>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %45, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_58 = cute.make_int_tuple() : () -> !cute.int_tuple<"5">
        %ptr_59 = cute.add_offset(%iter_5, %int_tuple_58) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"5">) -> !cute.ptr<i64, smem>
        %46 = builtin.unrealized_conversion_cast %ptr_59 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %46, %c1_i32 : !llvm.ptr<3>, i32
      }
      %int_tuple_6 = cute.make_int_tuple() : () -> !cute.int_tuple<"6">
      %ptr_7 = cute.add_offset(%iter_5, %int_tuple_6) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"6">) -> !cute.ptr<i64, smem, align<16>>
      scf.if %14 {
        %41 = builtin.unrealized_conversion_cast %ptr_7 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %41, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_50 = cute.make_int_tuple() : () -> !cute.int_tuple<"7">
        %ptr_51 = cute.add_offset(%iter_5, %int_tuple_50) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"7">) -> !cute.ptr<i64, smem>
        %42 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_52 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
        %ptr_53 = cute.add_offset(%iter_5, %int_tuple_52) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"8">) -> !cute.ptr<i64, smem, align<64>>
        %dyn = cute.derefine(%ptr_53) : !cute.ptr<i64, smem, align<64>> to !cute.ptr<i64, smem, align<16>>
        %43 = builtin.unrealized_conversion_cast %dyn : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_54 = cute.make_int_tuple() : () -> !cute.int_tuple<"9">
        %ptr_55 = cute.add_offset(%iter_5, %int_tuple_54) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"9">) -> !cute.ptr<i64, smem>
        %44 = builtin.unrealized_conversion_cast %ptr_55 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %44, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_56 = cute.make_int_tuple() : () -> !cute.int_tuple<"10">
        %ptr_57 = cute.add_offset(%iter_5, %int_tuple_56) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"10">) -> !cute.ptr<i64, smem, align<16>>
        %45 = builtin.unrealized_conversion_cast %ptr_57 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %45, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_58 = cute.make_int_tuple() : () -> !cute.int_tuple<"11">
        %ptr_59 = cute.add_offset(%iter_5, %int_tuple_58) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"11">) -> !cute.ptr<i64, smem>
        %46 = builtin.unrealized_conversion_cast %ptr_59 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %46, %c1_i32 : !llvm.ptr<3>, i32
      }
      %iter_8 = cute.recast_iter(%ptr_2) : !cute.ptr<i8, smem, align<32>> to !cute.ptr<i64, smem, align<32>>
      scf.if %14 {
        %41 = builtin.unrealized_conversion_cast %iter_8 : !cute.ptr<i64, smem, align<32>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %41, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_50 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
        %ptr_51 = cute.add_offset(%iter_8, %int_tuple_50) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
        %42 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
      }
      %int_tuple_9 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
      %ptr_10 = cute.add_offset(%iter_8, %int_tuple_9) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem, align<16>>
      scf.if %14 {
        %41 = builtin.unrealized_conversion_cast %ptr_10 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %41, %c4_i32 : !llvm.ptr<3>, i32
        %int_tuple_50 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
        %ptr_51 = cute.add_offset(%iter_8, %int_tuple_50) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
        %42 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c4_i32 : !llvm.ptr<3>, i32
      }
      nvvm.fence.mbarrier.init
      %15 = cute.composed_get_outer(%arg8) : (!cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">) -> !cute.layout<"(((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
      %16 = cute.ptrtoint(%ptr) : !cute.ptr<i8, smem, align<16>> to i32
      %17 = arith.addi %16, %c127_i32 : i32
      %18 = arith.andi %17, %c-128_i32 : i32
      %19 = arith.extsi %18 : i32 to i64
      %iv = cute.assume(%19) : (i64) -> !cute.i64<divby 128>
      %20 = cute.inttoptr(%iv) : !cute.i64<divby 128> to !cute.ptr<i8, smem, align<128>>
      %int_tuple_11 = cute.make_int_tuple() : () -> !cute.int_tuple<"98304">
      %ptr_12 = cute.add_offset(%20, %int_tuple_11) : (!cute.ptr<i8, smem, align<128>>, !cute.int_tuple<"98304">) -> !cute.ptr<i8, smem, align<128>>
      %21 = arith.cmpi sge, %smem_size, %c98560_i32 : i32
      cf.assert %21, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 98560 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_13 = cute.recast_iter(%20) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view = cute.make_view(%iter_13, %15) : !memref_smem_f16
      %22 = cute.composed_get_outer(%arg9) : (!cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">) -> !cute.layout<"(((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
      %int_tuple_14 = cute.make_int_tuple() : () -> !cute.int_tuple<"196608">
      %ptr_15 = cute.add_offset(%20, %int_tuple_14) : (!cute.ptr<i8, smem, align<128>>, !cute.int_tuple<"196608">) -> !cute.ptr<i8, smem, align<128>>
      %23 = arith.cmpi sge, %smem_size, %c196864_i32 : i32
      cf.assert %23, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 196864 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_16 = cute.recast_iter(%ptr_12) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view_17 = cute.make_view(%iter_16, %22) : !memref_smem_f16
      %tile_18 = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
      %coord = cute.make_coord() : () -> !cute.coord<"(_,_,_)">
      %tiled_view = cute.local_tile(%arg2, %tile_18, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">
      %tiled_view_19 = cute.local_tile(%arg4, %tile_18, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">
      %tile_20 = cute.make_tile() : () -> !cute.tile<"[128:1;128:1]">
      %tiled_view_21 = cute.local_tile(%arg6, %tile_20, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;128:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,128,?,?,?):(1@0,1@1,128@0,128@1,1@2)">
      %sz = cute.size(%tiled_view) <{mode = [3]}> : (!cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">) -> !cute.int_tuple<"?">
      %e0_22 = cute.get_leaves(%sz) : !cute.int_tuple<"?">
      %24 = cute.get_scalars(%e0_22) : !cute.int_tuple<"?">
      %coord_23 = cute.make_coord() : () -> !cute.coord<"0">
      %ptn_A = cute.tiled.mma.partition A (%arg0, %tiled_view, %coord_23) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">
      %ptn_B = cute.tiled.mma.partition B (%arg0, %tiled_view_19, %coord_23) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">
      %ptn_C = cute.tiled.mma.partition C (%arg0, %tiled_view_21, %coord_23) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,128,?,?,?):(1@0,1@1,128@0,128@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">
      %shape_24 = cute.make_shape() : () -> !cute.shape<"(1)">
      %lay_25 = cute.make_layout(%shape_24) : !cute.layout<"(1):(0)">
      %grouped = cute.group_modes(%view) <0, 3> : (!memref_smem_f16) -> !memref_smem_f16_1
      %grouped_26 = cute.group_modes(%ptn_A) <0, 3> : (!cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">
      %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg1, %coord_23, %lay_25, %grouped, %grouped_26) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"(1):(0)">, !memref_smem_f16_1, !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">) -> (!memref_smem_f16_2, !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">)
      %grouped_27 = cute.group_modes(%view_17) <0, 3> : (!memref_smem_f16) -> !memref_smem_f16_1
      %grouped_28 = cute.group_modes(%ptn_B) <0, 3> : (!cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">
      %res_smem_tensor_29, %res_target_tensors_30 = cute_nvgpu.atom.tma_partition(%arg3, %coord_23, %lay_25, %grouped_27, %grouped_28) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"(1):(0)">, !memref_smem_f16_1, !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">) -> (!memref_smem_f16_2, !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">)
      %frg_A = cute.mma.make_fragment A (%arg0, %view) : (!mma_f16_f16_f32_128x128x16, !memref_smem_f16) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">
      %frg_B = cute.mma.make_fragment B (%arg0, %view_17) : (!mma_f16_f16_f32_128x128x16, !memref_smem_f16) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">
      %shape_31 = cute.make_shape() : () -> !cute.shape<"((128,128),1,1,2)">
      %frg_C = cute.mma.make_fragment C (%arg0, %shape_31) : (!mma_f16_f16_f32_128x128x16, !cute.shape<"((128,128),1,1,2)">) -> !memref_tmem_f32
      nvvm.barrier
      %25 = nvvm.read.ptx.sreg.ctaid.z : i32
      %26 = nvvm.read.ptx.sreg.nctaid.x : i32
      %27 = nvvm.read.ptx.sreg.nctaid.y : i32
      %28 = nvvm.read.ptx.sreg.nctaid.z : i32
      %int_tuple_32 = cute.make_int_tuple(%26, %27, %28) : (i32, i32, i32) -> !cute.int_tuple<"(?,?,?)">
      %sz_33 = cute.size(%int_tuple_32) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
      %e0_34 = cute.get_leaves(%sz_33) : !cute.int_tuple<"?">
      %int_tuple_35 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
      %div = cute.tuple_div(%e0_34, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
      %29 = cute.get_scalars(%div) : !cute.int_tuple<"?">
      %sz_36 = cute.size(%lay) : (!cute.layout<"(?,?,?):(1,?,?)">) -> !cute.int_tuple<"?">
      %e0_37 = cute.get_leaves(%sz_36) : !cute.int_tuple<"?">
      %30 = cute.get_scalars(%e0_37) : !cute.int_tuple<"?">
      %31 = arith.cmpi sgt, %30, %25 : i32
      %quotient, %remainder = cute.fast_divmod.compute(%25, %arg16) : i32, !cute.fast_divmod_divisor<32> -> i32
      %quotient_38, %remainder_39 = cute.fast_divmod.compute(%quotient, %arg17) : i32, !cute.fast_divmod_divisor<32> -> i32
      %int_tuple_40 = cute.make_int_tuple(%remainder) : (i32) -> !cute.int_tuple<"?">
      %mul = cute.tuple_mul(%int_tuple_40, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
      %int_tuple_41 = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
      %add = cute.tuple_add(%mul, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
      %32 = cute.get_scalars(%add) : !cute.int_tuple<"?">
      %int_tuple_42 = cute.make_int_tuple(%remainder_39) : (i32) -> !cute.int_tuple<"?">
      %mul_43 = cute.tuple_mul(%int_tuple_42, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
      %add_44 = cute.tuple_add(%mul_43, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
      %33 = cute.get_scalars(%add_44) : !cute.int_tuple<"?">
      %int_tuple_45 = cute.make_int_tuple(%quotient_38) : (i32) -> !cute.int_tuple<"?">
      %mul_46 = cute.tuple_mul(%int_tuple_45, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
      %add_47 = cute.tuple_add(%mul_46, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
      %34 = cute.get_scalars(%add_47) : !cute.int_tuple<"?">
      %35:6 = scf.if %12 -> (i32, i32, i32, i1, i32, i32) {
        %41:8 = scf.while (%arg18 = %32, %arg19 = %33, %arg20 = %34, %arg21 = %31, %arg22 = %c0_i32, %arg23 = %c1_i32, %arg24 = %25, %arg25 = %c0_i32) : (i32, i32, i32, i1, i32, i32, i32, i32) -> (i32, i32, i32, i1, i32, i32, i32, i32) {
          scf.condition(%arg21) %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25 : i32, i32, i32, i1, i32, i32, i32, i32
        } do {
        ^bb0(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: i1, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32):
          %coord_52 = cute.make_coord(%arg18, %arg20) : (i32, i32) -> !cute.coord<"(_,?,_,?)">
          %slice = cute.slice(%res_target_tensors, %coord_52) : !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">, !cute.coord<"(_,?,_,?)">
          %coord_53 = cute.make_coord(%arg19, %arg20) : (i32, i32) -> !cute.coord<"(_,?,_,?)">
          %slice_54 = cute.slice(%res_target_tensors_30, %coord_53) : !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">, !cute.coord<"(_,?,_,?)">
          %int_tuple_55 = cute.make_int_tuple(%arg22) : (i32) -> !cute.int_tuple<"?">
          %ptr_56 = cute.add_offset(%ptr_7, %int_tuple_55) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %64 = builtin.unrealized_conversion_cast %ptr_56 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %65 = nvvm.mbarrier.wait.parity %64, %arg23 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
          %66:4 = scf.for %arg26 = %c0_i32 to %24 step %c1_i32 iter_args(%arg27 = %65, %arg28 = %c0_i32, %arg29 = %arg22, %arg30 = %arg23) -> (i1, i32, i32, i32)  : i32 {
            %73 = arith.extui %arg27 : i1 to i32
            %74 = arith.cmpi eq, %73, %c0_i32 : i32
            scf.if %74 {
              %int_tuple_109 = cute.make_int_tuple(%arg29) : (i32) -> !cute.int_tuple<"?">
              %ptr_110 = cute.add_offset(%ptr_7, %int_tuple_109) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %87 = builtin.unrealized_conversion_cast %ptr_110 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity.shared %87, %arg30, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %75 = nvvm.elect.sync -> i1
            scf.if %75 {
              %int_tuple_109 = cute.make_int_tuple(%arg29) : (i32) -> !cute.int_tuple<"?">
              %ptr_110 = cute.add_offset(%iter_5, %int_tuple_109) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %87 = builtin.unrealized_conversion_cast %ptr_110 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.txn %87, %c32768_i32 {kind = #nvvm.mbar_txn_kind<arrive_expect_tx>} : !llvm.ptr<3>, i32
            }
            %76 = arith.addi %arg29, %c1_i32 : i32
            %77 = arith.addi %arg28, %c1_i32 : i32
            %78 = arith.cmpi eq, %76, %c6_i32 : i32
            %79 = arith.select %78, %c0_i32, %76 : i32
            %80 = scf.if %78 -> (i32) {
              %87 = arith.xori %arg30, %c1_i32 : i32
              scf.yield %87 : i32
            } else {
              scf.yield %arg30 : i32
            }
            %coord_70 = cute.make_coord(%arg28) : (i32) -> !cute.coord<"(_,?)">
            %slice_71 = cute.slice(%slice, %coord_70) : !cute.coord_tensor<"(?{div=128},0,?)", "(((64,64),2),?):(((1@0,1@1),64@0),64@1)">, !cute.coord<"(_,?)">
            %iter_72 = cute.get_iter(%slice_71) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %tup = cute.deref_arith_tuple_iter(%iter_72) : !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %e0_73, %e1_74, %e2_75 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %coord_76 = cute.make_coord(%arg29) : (i32) -> !cute.coord<"(_,?)">
            %slice_77 = cute.slice(%res_smem_tensor, %coord_76) : !memref_smem_f16_2, !cute.coord<"(_,?)">
            %iter_78 = cute.get_iter(%slice_77) : !memref_smem_f16_3
            %int_tuple_79 = cute.make_int_tuple(%arg29) : (i32) -> !cute.int_tuple<"?">
            %ptr_80 = cute.add_offset(%iter_5, %int_tuple_79) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %lay_81 = cute.get_layout(%slice_71) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %shape_82 = cute.make_shape() : () -> !cute.shape<"1">
            %lay_83 = cute.make_layout(%shape_82) : !cute.layout<"1:0">
            %append = cute.append_to_rank<2> (%lay_81, %lay_83) : !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
            %int_tuple_84 = cute.make_int_tuple(%e0_73, %e1_74, %e2_75) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_84) : (!cute.int_tuple<"(?{div=128},?{div=64},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %view_85 = cute.make_view(%int_tup_iter, %append) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">
            %grouped_86 = cute.group_modes(%view_85) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">
            %lay_87 = cute.get_layout(%slice_77) : !memref_smem_f16_3
            %append_88 = cute.append_to_rank<2> (%lay_87, %lay_83) : !cute.layout<"((4096,2)):((1,4096))">, !cute.layout<"1:0">
            %view_89 = cute.make_view(%iter_78, %append_88) : !memref_smem_f16_4
            %grouped_90 = cute.group_modes(%view_89) <1, 2> : (!memref_smem_f16_4) -> !memref_smem_f16_5
            %81 = cute_nvgpu.atom.make_exec_tma(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>
            %82 = cute_nvgpu.atom.set_value<tma_bar>(%81, %ptr_80) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.ptr<i64, smem>)
            cute.copy(%82, %grouped_86, %grouped_90) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">, !memref_smem_f16_5)
            %slice_91 = cute.slice(%slice_54, %coord_70) : !cute.coord_tensor<"(?{div=128},0,?)", "(((64,64),2),?):(((1@0,1@1),64@0),64@1)">, !cute.coord<"(_,?)">
            %iter_92 = cute.get_iter(%slice_91) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %tup_93 = cute.deref_arith_tuple_iter(%iter_92) : !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %e0_94, %e1_95, %e2_96 = cute.get_leaves(%tup_93) : !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %slice_97 = cute.slice(%res_smem_tensor_29, %coord_76) : !memref_smem_f16_2, !cute.coord<"(_,?)">
            %iter_98 = cute.get_iter(%slice_97) : !memref_smem_f16_3
            %lay_99 = cute.get_layout(%slice_91) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %append_100 = cute.append_to_rank<2> (%lay_99, %lay_83) : !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
            %int_tuple_101 = cute.make_int_tuple(%e0_94, %e1_95, %e2_96) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %int_tup_iter_102 = cute.make_arith_tuple_iter(%int_tuple_101) : (!cute.int_tuple<"(?{div=128},?{div=64},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %view_103 = cute.make_view(%int_tup_iter_102, %append_100) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">
            %grouped_104 = cute.group_modes(%view_103) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">
            %lay_105 = cute.get_layout(%slice_97) : !memref_smem_f16_3
            %append_106 = cute.append_to_rank<2> (%lay_105, %lay_83) : !cute.layout<"((4096,2)):((1,4096))">, !cute.layout<"1:0">
            %view_107 = cute.make_view(%iter_98, %append_106) : !memref_smem_f16_4
            %grouped_108 = cute.group_modes(%view_107) <1, 2> : (!memref_smem_f16_4) -> !memref_smem_f16_5
            %83 = cute_nvgpu.atom.make_exec_tma(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>
            %84 = cute_nvgpu.atom.set_value<tma_bar>(%83, %ptr_80) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.ptr<i64, smem>)
            cute.copy(%84, %grouped_104, %grouped_108) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">, !memref_smem_f16_5)
            %85 = arith.cmpi sgt, %24, %77 : i32
            %86 = scf.if %85 -> (i1) {
              %int_tuple_109 = cute.make_int_tuple(%79) : (i32) -> !cute.int_tuple<"?">
              %ptr_110 = cute.add_offset(%ptr_7, %int_tuple_109) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %87 = builtin.unrealized_conversion_cast %ptr_110 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %88 = nvvm.mbarrier.wait.parity %87, %80 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %88 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %86, %77, %79, %80 : i1, i32, i32, i32
          } {loop_annotation = #loop_annotation}
          %67 = arith.addi %arg24, %29 : i32
          %68 = arith.addi %arg25, %c1_i32 : i32
          %69 = arith.cmpi sgt, %30, %67 : i32
          %quotient_57, %remainder_58 = cute.fast_divmod.compute(%67, %arg16) : i32, !cute.fast_divmod_divisor<32> -> i32
          %quotient_59, %remainder_60 = cute.fast_divmod.compute(%quotient_57, %arg17) : i32, !cute.fast_divmod_divisor<32> -> i32
          %int_tuple_61 = cute.make_int_tuple(%remainder_58) : (i32) -> !cute.int_tuple<"?">
          %mul_62 = cute.tuple_mul(%int_tuple_61, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_63 = cute.tuple_add(%mul_62, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %70 = cute.get_scalars(%add_63) : !cute.int_tuple<"?">
          %int_tuple_64 = cute.make_int_tuple(%remainder_60) : (i32) -> !cute.int_tuple<"?">
          %mul_65 = cute.tuple_mul(%int_tuple_64, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_66 = cute.tuple_add(%mul_65, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %71 = cute.get_scalars(%add_66) : !cute.int_tuple<"?">
          %int_tuple_67 = cute.make_int_tuple(%quotient_59) : (i32) -> !cute.int_tuple<"?">
          %mul_68 = cute.tuple_mul(%int_tuple_67, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_69 = cute.tuple_add(%mul_68, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %72 = cute.get_scalars(%add_69) : !cute.int_tuple<"?">
          scf.yield %70, %71, %72, %69, %66#2, %66#3, %67, %68 : i32, i32, i32, i1, i32, i32, i32, i32
        }
        %42 = arith.addi %41#4, %c1_i32 : i32
        %43 = arith.cmpi eq, %42, %c6_i32 : i32
        %44 = arith.select %43, %c0_i32, %42 : i32
        %45 = scf.if %43 -> (i32) {
          %64 = arith.xori %41#5, %c1_i32 : i32
          scf.yield %64 : i32
        } else {
          scf.yield %41#5 : i32
        }
        %46 = arith.addi %44, %c1_i32 : i32
        %47 = arith.cmpi eq, %46, %c6_i32 : i32
        %48 = arith.select %47, %c0_i32, %46 : i32
        %49 = scf.if %47 -> (i32) {
          %64 = arith.xori %45, %c1_i32 : i32
          scf.yield %64 : i32
        } else {
          scf.yield %45 : i32
        }
        %50 = arith.addi %48, %c1_i32 : i32
        %51 = arith.cmpi eq, %50, %c6_i32 : i32
        %52 = arith.select %51, %c0_i32, %50 : i32
        %53 = scf.if %51 -> (i32) {
          %64 = arith.xori %49, %c1_i32 : i32
          scf.yield %64 : i32
        } else {
          scf.yield %49 : i32
        }
        %54 = arith.addi %52, %c1_i32 : i32
        %55 = arith.cmpi eq, %54, %c6_i32 : i32
        %56 = arith.select %55, %c0_i32, %54 : i32
        %57 = scf.if %55 -> (i32) {
          %64 = arith.xori %53, %c1_i32 : i32
          scf.yield %64 : i32
        } else {
          scf.yield %53 : i32
        }
        %58 = arith.addi %56, %c1_i32 : i32
        %59 = arith.cmpi eq, %58, %c6_i32 : i32
        %60 = arith.select %59, %c0_i32, %58 : i32
        %61 = scf.if %59 -> (i32) {
          %64 = arith.xori %57, %c1_i32 : i32
          scf.yield %64 : i32
        } else {
          scf.yield %57 : i32
        }
        %int_tuple_50 = cute.make_int_tuple(%60) : (i32) -> !cute.int_tuple<"?">
        %ptr_51 = cute.add_offset(%ptr_7, %int_tuple_50) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
        %62 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.try_wait.parity.shared %62, %61, %c10000000_i32 : !llvm.ptr<3>, i32, i32
        %63 = nvvm.elect.sync -> i1
        scf.if %63 {
          %ptr_52 = cute.add_offset(%iter_5, %int_tuple_50) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %64 = builtin.unrealized_conversion_cast %ptr_52 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.txn %64, %c32768_i32 {kind = #nvvm.mbar_txn_kind<arrive_expect_tx>} : !llvm.ptr<3>, i32
        }
        scf.yield %41#0, %41#1, %41#2, %41#3, %41#6, %41#7 : i32, i32, i32, i1, i32, i32
      } else {
        scf.yield %32, %33, %34, %31, %25, %c0_i32 : i32, i32, i32, i1, i32, i32
      }
      %36 = arith.cmpi eq, %11, %c4_i32 : i32
      %37:6 = scf.if %36 -> (i32, i32, i32, i1, i32, i32) {
        nvvm.barrier id = %c2_i32 number_of_threads = %c160_i32
        %tmem_ptr = cute_nvgpu.arch.sm100.retrieve_tmem_ptr(%iter) : !cute.ptr<i32, smem, align<8>> -> !cute.ptr<f32, tmem, align<16>>
        %lay_50 = cute.get_layout(%frg_C) : !memref_tmem_f32
        %view_51 = cute.make_view(%tmem_ptr, %lay_50) : !memref_tmem_f32_1
        %41:12 = scf.while (%arg18 = %35#0, %arg19 = %35#1, %arg20 = %35#2, %arg21 = %35#3, %arg22 = %c0_i32, %arg23 = %c0_i32, %arg24 = %arg0, %arg25 = %c0_i32, %arg26 = %c0_i32, %arg27 = %c1_i32, %arg28 = %35#4, %arg29 = %35#5) : (i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32) -> (i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32) {
          scf.condition(%arg21) %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29 : i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: i1, %arg22: i32, %arg23: i32, %arg24: !mma_f16_f16_f32_128x128x16, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32):
          %coord_52 = cute.make_coord(%arg26) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice = cute.slice(%view_51, %coord_52) : !memref_tmem_f32_1, !cute.coord<"(_,_,_,?)">
          %int_tuple_53 = cute.make_int_tuple(%arg22) : (i32) -> !cute.int_tuple<"?">
          %ptr_54 = cute.add_offset(%iter_5, %int_tuple_53) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %46 = builtin.unrealized_conversion_cast %ptr_54 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %47 = nvvm.mbarrier.wait.parity %46, %arg23 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
          %int_tuple_55 = cute.make_int_tuple(%arg26) : (i32) -> !cute.int_tuple<"?">
          %ptr_56 = cute.add_offset(%ptr_10, %int_tuple_55) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %48 = builtin.unrealized_conversion_cast %ptr_56 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %48, %arg27, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %49 = cute_nvgpu.atom.set_value<accum_c>(%arg24, %false) : (!mma_f16_f16_f32_128x128x16, i1)
          %50:5 = scf.for %arg30 = %c0_i32 to %24 step %c1_i32 iter_args(%arg31 = %47, %arg32 = %c0_i32, %arg33 = %arg22, %arg34 = %arg23, %arg35 = %49) -> (i1, i32, i32, i32, !mma_f16_f16_f32_128x128x16)  : i32 {
            %63 = arith.extui %arg31 : i1 to i32
            %64 = arith.cmpi eq, %63, %c0_i32 : i32
            scf.if %64 {
              %int_tuple_70 = cute.make_int_tuple(%arg33) : (i32) -> !cute.int_tuple<"?">
              %ptr_71 = cute.add_offset(%iter_5, %int_tuple_70) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %74 = builtin.unrealized_conversion_cast %ptr_71 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity.shared %74, %arg34, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %65 = arith.addi %arg33, %c1_i32 : i32
            %66 = arith.addi %arg32, %c1_i32 : i32
            %67 = arith.cmpi eq, %65, %c6_i32 : i32
            %68 = arith.select %67, %c0_i32, %65 : i32
            %69 = scf.if %67 -> (i32) {
              %74 = arith.xori %arg34, %c1_i32 : i32
              scf.yield %74 : i32
            } else {
              scf.yield %arg34 : i32
            }
            %70 = scf.for %arg36 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg37 = %arg35) -> (!mma_f16_f16_f32_128x128x16)  : i32 {
              %coord_70 = cute.make_coord(%arg36, %arg33) : (i32, i32) -> !cute.coord<"(_,_,?,?)">
              %slice_71 = cute.slice(%frg_A, %coord_70) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">, !cute.coord<"(_,_,?,?)">
              %slice_72 = cute.slice(%frg_B, %coord_70) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">, !cute.coord<"(_,_,?,?)">
              cute.gemm(%arg37, %slice, %slice_71, %slice_72, %slice) : (!mma_f16_f16_f32_128x128x16, !memref_tmem_f32_2, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_tmem_f32_2)
              %74 = cute_nvgpu.atom.set_value<accum_c>(%arg37, %true) : (!mma_f16_f16_f32_128x128x16, i1)
              scf.yield %74 : !mma_f16_f16_f32_128x128x16
            } {loop_annotation = #loop_annotation1}
            %71 = nvvm.elect.sync -> i1
            scf.if %71 {
              %int_tuple_70 = cute.make_int_tuple(%arg33) : (i32) -> !cute.int_tuple<"?">
              %ptr_71 = cute.add_offset(%ptr_7, %int_tuple_70) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %74 = builtin.unrealized_conversion_cast %ptr_71 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.tcgen05.commit %74 : !llvm.ptr<3>
            }
            %72 = arith.cmpi sgt, %24, %66 : i32
            %73 = scf.if %72 -> (i1) {
              %int_tuple_70 = cute.make_int_tuple(%68) : (i32) -> !cute.int_tuple<"?">
              %ptr_71 = cute.add_offset(%iter_5, %int_tuple_70) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %74 = builtin.unrealized_conversion_cast %ptr_71 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %75 = nvvm.mbarrier.wait.parity %74, %69 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %75 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %73, %66, %68, %69, %70 : i1, i32, i32, i32, !mma_f16_f16_f32_128x128x16
          }
          %51 = nvvm.elect.sync -> i1
          scf.if %51 {
            %ptr_70 = cute.add_offset(%iter_8, %int_tuple_55) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %63 = builtin.unrealized_conversion_cast %ptr_70 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.tcgen05.commit %63 : !llvm.ptr<3>
          }
          %52 = arith.addi %arg26, %c1_i32 : i32
          %53 = arith.addi %arg25, %c1_i32 : i32
          %54 = arith.cmpi eq, %52, %c2_i32 : i32
          %55 = arith.select %54, %c0_i32, %52 : i32
          %56 = scf.if %54 -> (i32) {
            %63 = arith.xori %arg27, %c1_i32 : i32
            scf.yield %63 : i32
          } else {
            scf.yield %arg27 : i32
          }
          %57 = arith.addi %arg28, %29 : i32
          %58 = arith.addi %arg29, %c1_i32 : i32
          %59 = arith.cmpi sgt, %30, %57 : i32
          %quotient_57, %remainder_58 = cute.fast_divmod.compute(%57, %arg16) : i32, !cute.fast_divmod_divisor<32> -> i32
          %quotient_59, %remainder_60 = cute.fast_divmod.compute(%quotient_57, %arg17) : i32, !cute.fast_divmod_divisor<32> -> i32
          %int_tuple_61 = cute.make_int_tuple(%remainder_58) : (i32) -> !cute.int_tuple<"?">
          %mul_62 = cute.tuple_mul(%int_tuple_61, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_63 = cute.tuple_add(%mul_62, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %60 = cute.get_scalars(%add_63) : !cute.int_tuple<"?">
          %int_tuple_64 = cute.make_int_tuple(%remainder_60) : (i32) -> !cute.int_tuple<"?">
          %mul_65 = cute.tuple_mul(%int_tuple_64, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_66 = cute.tuple_add(%mul_65, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %61 = cute.get_scalars(%add_66) : !cute.int_tuple<"?">
          %int_tuple_67 = cute.make_int_tuple(%quotient_59) : (i32) -> !cute.int_tuple<"?">
          %mul_68 = cute.tuple_mul(%int_tuple_67, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_69 = cute.tuple_add(%mul_68, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %62 = cute.get_scalars(%add_69) : !cute.int_tuple<"?">
          scf.yield %60, %61, %62, %59, %50#2, %50#3, %50#4, %53, %55, %56, %57, %58 : i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32
        }
        %42 = nvvm.read.ptx.sreg.cluster.ctarank : i32
        %43 = cute_nvgpu.arch.make_warp_uniform(%42) : i32
        %44 = arith.remsi %43, %c2_i32 : i32
        %45 = arith.cmpi eq, %44, %c0_i32 : i32
        scf.if %45 {
          %46 = arith.addi %41#8, %c1_i32 : i32
          %47 = arith.cmpi eq, %46, %c2_i32 : i32
          %48 = arith.select %47, %c0_i32, %46 : i32
          %49 = scf.if %47 -> (i32) {
            %51 = arith.xori %41#9, %c1_i32 : i32
            scf.yield %51 : i32
          } else {
            scf.yield %41#9 : i32
          }
          %int_tuple_52 = cute.make_int_tuple(%48) : (i32) -> !cute.int_tuple<"?">
          %ptr_53 = cute.add_offset(%ptr_10, %int_tuple_52) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %50 = builtin.unrealized_conversion_cast %ptr_53 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %50, %49, %c10000000_i32 : !llvm.ptr<3>, i32, i32
        }
        scf.yield %41#0, %41#1, %41#2, %41#3, %41#10, %41#11 : i32, i32, i32, i1, i32, i32
      } else {
        scf.yield %35#0, %35#1, %35#2, %35#3, %35#4, %35#5 : i32, i32, i32, i1, i32, i32
      }
      %38 = cute.composed_get_outer(%arg10) : (!cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">) -> !cute.layout<"((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">
      %39 = arith.cmpi sge, %smem_size, %c229632_i32 : i32
      cf.assert %39, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 229632 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_48 = cute.recast_iter(%ptr_15) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view_49 = cute.make_view(%iter_48, %38) : !memref_smem_f16_6
      %40 = arith.cmpi slt, %11, %c4_i32 : i32
      scf.if %40 {
        scf.if %14 {
          cute_nvgpu.arch.sm100.alloc_tmem(%c256_i32, %iter) [ cta_1] : i32, !cute.ptr<i32, smem, align<8>>
        }
        nvvm.barrier id = %c2_i32 number_of_threads = %c160_i32
        %tmem_ptr = cute_nvgpu.arch.sm100.retrieve_tmem_ptr(%iter) : !cute.ptr<i32, smem, align<8>> -> !cute.ptr<f32, tmem, align<16>>
        %41:8 = scf.while (%arg18 = %37#0, %arg19 = %37#1, %arg20 = %37#2, %arg21 = %37#3, %arg22 = %c0_i32, %arg23 = %c0_i32, %arg24 = %c0_i32, %arg25 = %37#4, %arg26 = %37#5) : (i32, i32, i32, i1, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg21) %arg18, %arg19, %arg20, %arg22, %arg23, %arg24, %arg25, %arg26 : i32, i32, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32):
          %42 = arith.addi %arg24, %29 : i32
          %43 = arith.addi %arg25, %c1_i32 : i32
          %44 = arith.cmpi sgt, %30, %42 : i32
          %quotient_50, %remainder_51 = cute.fast_divmod.compute(%42, %arg16) : i32, !cute.fast_divmod_divisor<32> -> i32
          %quotient_52, %remainder_53 = cute.fast_divmod.compute(%quotient_50, %arg17) : i32, !cute.fast_divmod_divisor<32> -> i32
          %int_tuple_54 = cute.make_int_tuple(%remainder_51) : (i32) -> !cute.int_tuple<"?">
          %mul_55 = cute.tuple_mul(%int_tuple_54, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_56 = cute.tuple_add(%mul_55, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %45 = cute.get_scalars(%add_56) : !cute.int_tuple<"?">
          %int_tuple_57 = cute.make_int_tuple(%remainder_53) : (i32) -> !cute.int_tuple<"?">
          %mul_58 = cute.tuple_mul(%int_tuple_57, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_59 = cute.tuple_add(%mul_58, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %46 = cute.get_scalars(%add_59) : !cute.int_tuple<"?">
          %int_tuple_60 = cute.make_int_tuple(%quotient_52) : (i32) -> !cute.int_tuple<"?">
          %mul_61 = cute.tuple_mul(%int_tuple_60, %int_tuple_35) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
          %add_62 = cute.tuple_add(%mul_61, %int_tuple_41) : (!cute.int_tuple<"?">, !cute.int_tuple<"0">) -> !cute.int_tuple<"?">
          %47 = cute.get_scalars(%add_62) : !cute.int_tuple<"?">
          %lay_63 = cute.get_layout(%ptn_C) : !cute.coord_tensor<"(0,0,0)", "((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">
          %48 = cute.get_shape(%lay_63) : (!cute.layout<"((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">) -> !cute.shape<"((128,128),1,1,?,?,?)">
          %e0_64, %e1_65, %e2_66, %e3, %e4, %e5, %e6 = cute.get_leaves(%48) : !cute.shape<"((128,128),1,1,?,?,?)">
          %itup = cute.to_int_tuple(%e4) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %itup_67 = cute.to_int_tuple(%e5) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %itup_68 = cute.to_int_tuple(%e6) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %shape_69 = cute.make_shape(%itup, %itup_67, %itup_68) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"((128,1),(128,1),?,?,?)">
          %stride = cute.make_stride() : () -> !cute.stride<"((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %lay_70 = cute.make_layout(%shape_69, %stride) : !cute.layout<"((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %int_tuple_71 = cute.make_int_tuple() : () -> !cute.int_tuple<"(0,0,0)">
          %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_71) : (!cute.int_tuple<"(0,0,0)">) -> !cute.arith_tuple_iter<"(0,0,0)">
          %view_72 = cute.make_view(%int_tup_iter, %lay_70) : !cute.coord_tensor<"(0,0,0)", "((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %shape_73 = cute.make_shape() : () -> !cute.shape<"((128,1),(128,1),2)">
          %stride_74 = cute.make_stride() : () -> !cute.stride<"((65536,0),(1,0),128)">
          %lay_75 = cute.make_layout(%shape_73, %stride_74) : !cute.layout<"((128,1),(128,1),2):((65536,0),(1,0),128)">
          %view_76 = cute.make_view(%tmem_ptr, %lay_75) : !memref_tmem_f32_3
          %atom = cute.make_atom() : () -> !cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x4>
          %tile_77 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
          %div_78 = cute.flat_divide(%view_76, %tile_77) : !memref_tmem_f32_3, !cute.tile<"[128:1;32:1]">
          %coord_79 = cute.make_coord() : () -> !cute.coord<"(_,_,0,0,0)">
          %slice = cute.slice(%div_78, %coord_79) : !memref_tmem_f32_4, !cute.coord<"(_,_,0,0,0)">
          %49 = cute_nvgpu.atom.make_tmem_copy(%atom, %slice) : (!cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x4>, !memref_tmem_f32_5) -> !copy_ldtm_256
          %coord_80 = cute.make_coord(%0) : (i32) -> !cute.coord<"?">
          %src_partitioned = cute.tiled.copy.partition_S(%49, %div_78, %coord_80) : (!copy_ldtm_256, !memref_tmem_f32_4, !cute.coord<"?">) -> !memref_tmem_f32_6
          %rmem = cute.memref.alloca() : !memref_rmem_f32
          %iter_81 = cute.get_iter(%rmem) : !memref_rmem_f32
          %rmem_82 = cute.memref.alloca() : !memref_rmem_f16
          %atom_83 = cute.make_atom() : () -> !cute_nvgpu.atom.stsm<f16, mode = <"(8,8)">, num_matrices = 4, t>
          %50 = cute.make_tiled_copy(%atom_83) : !copy_stsm_4
          %dst_partitioned = cute.tiled.copy.partition_D(%50, %view_49, %coord_80) : (!copy_stsm_4, !memref_smem_f16_6, !cute.coord<"?">) -> !memref_smem_f16_7
          %retiled = cute.tiled.copy.retile(%50, %rmem_82) : (!copy_stsm_4, !memref_rmem_f16) -> !memref_rmem_f16_1
          %div_84 = cute.flat_divide(%view_72, %tile_77) : !cute.coord_tensor<"(0,0,0)", "((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">, !cute.tile<"[128:1;32:1]">
          %shape_85 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_86 = cute.make_layout(%shape_85) : !cute.layout<"1:0">
          %grouped_87 = cute.group_modes(%view_49) <0, 2> : (!memref_smem_f16_6) -> !memref_smem_f16_8
          %grouped_88 = cute.group_modes(%div_84) <0, 2> : (!cute.coord_tensor<"(0,0,0)", "(128,32,1,4,?,?,?):(1@0,1@1,0,32@1,128@0,128@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "((128,32),1,4,?,?,?):((1@0,1@1),0,32@1,128@0,128@1,1@2)">
          %res_smem_tensor_89, %res_target_tensors_90 = cute_nvgpu.atom.tma_partition(%arg5, %coord_23, %lay_86, %grouped_87, %grouped_88) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_f16_8, !cute.coord_tensor<"(0,0,0)", "((128,32),1,4,?,?,?):((1@0,1@1),0,32@1,128@0,128@1,1@2)">) -> (!memref_smem_f16_9, !cute.coord_tensor<"(0,0,0)", "(((64,32),2),1,4,?,?,?):(((1@0,1@1),64@0),0,32@1,128@0,128@1,1@2)">)
          %coord_91 = cute.make_coord(%arg18, %arg19, %arg20) : (i32, i32, i32) -> !cute.coord<"(_,_,_,?,?,?)">
          %slice_92 = cute.slice(%res_target_tensors_90, %coord_91) : !cute.coord_tensor<"(0,0,0)", "(((64,32),2),1,4,?,?,?):(((1@0,1@1),64@0),0,32@1,128@0,128@1,1@2)">, !cute.coord<"(_,_,_,?,?,?)">
          %coord_93 = cute.make_coord(%arg22) : (i32) -> !cute.coord<"(_,_,_,_,_,?)">
          %slice_94 = cute.slice(%src_partitioned, %coord_93) : !memref_tmem_f32_6, !cute.coord<"(_,_,_,_,_,?)">
          %int_tuple_95 = cute.make_int_tuple(%arg22) : (i32) -> !cute.int_tuple<"?">
          %ptr_96 = cute.add_offset(%iter_8, %int_tuple_95) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %51 = builtin.unrealized_conversion_cast %ptr_96 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %51, %arg23, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %grouped_97 = cute.group_modes(%slice_94) <3, 5> : (!memref_tmem_f32_7) -> !memref_tmem_f32_8
          %grouped_98 = cute.group_modes(%slice_92) <1, 3> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),1,4):(((1@0,1@1),64@0),0,32@1)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),(1,4)):(((1@0,1@1),64@0),(0,32@1))">
          %52 = arith.muli %43, %c4_i32 : i32
          scf.for %arg26 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
            %iter_99 = cute.get_iter(%retiled) : !memref_rmem_f16_1
            %coord_100 = cute.make_coord(%arg26) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_101 = cute.slice(%grouped_97, %coord_100) : !memref_tmem_f32_8, !cute.coord<"(_,_,_,?)">
            %iter_102 = cute.get_iter(%slice_101) : !memref_tmem_f32_9
            %lay_103 = cute.get_layout(%slice_101) : !memref_tmem_f32_9
            %append = cute.append_to_rank<2> (%lay_103, %lay_86) : !cute.layout<"(((32,16),1),2,1):(((1,65536),0),1048576,0)">, !cute.layout<"1:0">
            %view_104 = cute.make_view(%iter_102, %append) : !memref_tmem_f32_9
            %grouped_105 = cute.group_modes(%view_104) <1, 3> : (!memref_tmem_f32_9) -> !memref_tmem_f32_10
            %lay_106 = cute.get_layout(%rmem) : !memref_rmem_f32
            %append_107 = cute.append_to_rank<2> (%lay_106, %lay_86) : !cute.layout<"(((2,2,4),1),2,1):(((1,2,4),0),16,0)">, !cute.layout<"1:0">
            %view_108 = cute.make_view(%iter_81, %append_107) : !memref_rmem_f32
            %grouped_109 = cute.group_modes(%view_108) <1, 3> : (!memref_rmem_f32) -> !memref_rmem_f32_1
            cute.copy(%49, %grouped_105, %grouped_109) : (!copy_ldtm_256, !memref_tmem_f32_10, !memref_rmem_f32_1)
            %retiled_110 = cute.tiled.copy.retile(%50, %rmem) : (!copy_stsm_4, !memref_rmem_f32) -> !memref_rmem_f32_2
            %59 = cute.memref.load_vec(%retiled_110) : (!memref_rmem_f32_2) -> vector<32xf32>
            %60 = arith.truncf %59 : vector<32xf32> to vector<32xf16>
            cute.memref.store_vec(%60, %retiled) : (vector<32xf16>, !memref_rmem_f16_1) -> ()
            %61 = arith.addi %52, %arg26 : i32
            %62 = arith.remsi %61, %c4_i32 : i32
            %coord_111 = cute.make_coord(%62) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_112 = cute.slice(%dst_partitioned, %coord_111) : !memref_smem_f16_7, !cute.coord<"(_,_,_,?)">
            %iter_113 = cute.get_iter(%slice_112) : !memref_smem_f16_10
            %lay_114 = cute.get_layout(%retiled) : !memref_rmem_f16_1
            %append_115 = cute.append_to_rank<2> (%lay_114, %lay_86) : !cute.layout<"((8,2),2,1):((1,8),16,0)">, !cute.layout<"1:0">
            %view_116 = cute.make_view(%iter_99, %append_115) : !memref_rmem_f16_1
            %grouped_117 = cute.group_modes(%view_116) <1, 3> : (!memref_rmem_f16_1) -> !memref_rmem_f16_2
            %lay_118 = cute.get_layout(%slice_112) : !memref_smem_f16_10
            %append_119 = cute.append_to_rank<2> (%lay_118, %lay_86) : !cute.layout<"((8,2),2,1):((1,1024),16,0)">, !cute.layout<"1:0">
            %view_120 = cute.make_view(%iter_113, %append_119) : !memref_smem_f16_10
            %grouped_121 = cute.group_modes(%view_120) <1, 3> : (!memref_smem_f16_10) -> !memref_smem_f16_11
            cute.copy(%50, %grouped_117, %grouped_121) : (!copy_stsm_4, !memref_rmem_f16_2, !memref_smem_f16_11)
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
            scf.if %14 {
              %coord_122 = cute.make_coord(%62) : (i32) -> !cute.coord<"(_,?)">
              %slice_123 = cute.slice(%res_smem_tensor_89, %coord_122) : !memref_smem_f16_9, !cute.coord<"(_,?)">
              %iter_124 = cute.get_iter(%slice_123) : !memref_smem_f16_12
              %coord_125 = cute.make_coord(%arg26) : (i32) -> !cute.coord<"(_,?)">
              %slice_126 = cute.slice(%grouped_98, %coord_125) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),(1,4)):(((1@0,1@1),64@0),(0,32@1))">, !cute.coord<"(_,?)">
              %iter_127 = cute.get_iter(%slice_126) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2)):(((1@0,1@1),64@0))">
              %tup = cute.deref_arith_tuple_iter(%iter_127) : !cute.arith_tuple_iter<"(?{div=128},?{div=32},?)">
              %e0_128, %e1_129, %e2_130 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=32},?)">
              %lay_131 = cute.get_layout(%slice_123) : !memref_smem_f16_12
              %append_132 = cute.append_to_rank<2> (%lay_131, %lay_86) : !cute.layout<"((2048,2)):((1,2048))">, !cute.layout<"1:0">
              %view_133 = cute.make_view(%iter_124, %append_132) : !memref_smem_f16_13
              %grouped_134 = cute.group_modes(%view_133) <1, 2> : (!memref_smem_f16_13) -> !memref_smem_f16_14
              %lay_135 = cute.get_layout(%slice_126) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2)):(((1@0,1@1),64@0))">
              %append_136 = cute.append_to_rank<2> (%lay_135, %lay_86) : !cute.layout<"(((64,32),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
              %int_tuple_137 = cute.make_int_tuple(%e0_128, %e1_129, %e2_130) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=32},?)">
              %int_tup_iter_138 = cute.make_arith_tuple_iter(%int_tuple_137) : (!cute.int_tuple<"(?{div=128},?{div=32},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=32},?)">
              %view_139 = cute.make_view(%int_tup_iter_138, %append_136) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),1):(((1@0,1@1),64@0),0)">
              %grouped_140 = cute.group_modes(%view_139) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),(1)):(((1@0,1@1),64@0),(0))">
              %63 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_store<f16, copy_bits = 32768, mode = tiled, g_stride = <"()"> tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">>
              cute.copy(%63, %grouped_134, %grouped_140) : (!cute_nvgpu.atom.tma_store<f16, copy_bits = 32768, mode = tiled, g_stride = <"()"> tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">>, !memref_smem_f16_14, !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),(1)):(((1@0,1@1),64@0),(0))">)
              nvvm.cp.async.bulk.commit.group
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
          %53 = nvvm.elect.sync -> i1
          scf.if %53 {
            %ptr_99 = cute.add_offset(%ptr_10, %int_tuple_95) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %59 = builtin.unrealized_conversion_cast %ptr_99 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.txn %59, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>} : !llvm.ptr<3>, i32
          }
          %54 = arith.addi %arg22, %c1_i32 : i32
          %55 = arith.addi %arg21, %c1_i32 : i32
          %56 = arith.cmpi eq, %54, %c2_i32 : i32
          %57 = arith.select %56, %c0_i32, %54 : i32
          %58 = scf.if %56 -> (i32) {
            %59 = arith.xori %arg23, %c1_i32 : i32
            scf.yield %59 : i32
          } else {
            scf.yield %arg23 : i32
          }
          scf.yield %45, %46, %47, %44, %55, %57, %58, %42, %43 : i32, i32, i32, i1, i32, i32, i32, i32, i32
        }
        nvvm.cp.async.bulk.wait_group 0 {read}
        scf.if %14 {
          cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit [ cta_1]
        }
        scf.if %14 {
          cute_nvgpu.arch.sm100.dealloc_tmem(%tmem_ptr, %c256_i32) [ cta_1] : !cute.ptr<f32, tmem, align<16>>, i32
        }
      }
      return
    }
  }
  func.func @cutlass_bmm_infraswe_b200_static_replay_2PersistentDenseGemmKernelobjectat_Tensorgmemoi641i64_Tensorgmemoi64i641_Tensorgmemoi641i64_148_FakeStream_functionrunlocalslambdaat(%arg0: !memref_gmem_f16, %arg1: !memref_gmem_f16_1, %arg2: !memref_gmem_f16, %arg3: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c229632_i64 = arith.constant 229632 : i64
    %c0_i32 = arith.constant 0 : i32
    %c192_i32 = arith.constant 192 : i32
    %c1_i32 = arith.constant 1 : i32
    %c148_i32 = arith.constant 148 : i32
    %0 = cute.static : !cute.layout<"((1,(1,1)),((128,16),(1,4))):((1@1,(0,0)),((1@0,1@1),(0,16@1)))">
    %1 = cute.static : !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,(1,6)):(((1,4096),64),0,1024,(0,8192))">
    %2 = cute.static : !cute.swizzle<"S<3,4,3>">
    %false = arith.constant false
    %iter = cute.get_iter(%arg0) : !memref_gmem_f16
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f16_1
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f16
    %lay = cute.get_layout(%arg0) : !memref_gmem_f16
    %3 = cute.select<[1, 2, 0]> (%lay) : (!cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.layout<"(?,?,?):(1,?{i64},?{i64})">
    %view = cute.make_view(%iter, %3) : !memref_gmem_f16_2
    %lay_2 = cute.get_layout(%arg1) : !memref_gmem_f16_1
    %4 = cute.select<[2, 1, 0]> (%lay_2) : (!cute.layout<"(?,?,?):(?{i64},?{i64},1)">) -> !cute.layout<"(?,?,?):(1,?{i64},?{i64})">
    %view_3 = cute.make_view(%iter_0, %4) : !memref_gmem_f16_2
    %lay_4 = cute.get_layout(%arg2) : !memref_gmem_f16
    %5 = cute.select<[1, 2, 0]> (%lay_4) : (!cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.layout<"(?,?,?):(1,?{i64},?{i64})">
    %view_5 = cute.make_view(%iter_1, %5) : !memref_gmem_f16_2
    %atom = cute.make_atom(%false, %false, %false) : (i1, i1, i1) -> !cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (mn, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0>
    %6 = cute.make_tiled_mma(%atom) : !mma_f16_f16_f32_128x128x16
    %shape = cute.make_shape() : () -> !cute.shape<"(1,1,1)">
    %lay_6 = cute.make_layout(%shape) : !cute.layout<"(1,1,1):(0,0,0)">
    %tile = cute.make_tile() : () -> !cute.tile<"[1:0]">
    %div = cute.tiled_divide(%lay_6, %tile) : !cute.layout<"(1,1,1):(0,0,0)">, !cute.tile<"[1:0]">
    %shape_7 = cute.make_shape() : () -> !cute.shape<"128">
    %lay_8 = cute.make_layout(%shape_7) : !cute.layout<"128:1">
    %shape_9 = cute.make_shape() : () -> !cute.shape<"(32,1)">
    %stride = cute.make_stride() : () -> !cute.stride<"(1,128)">
    %lay_10 = cute.make_layout(%shape_9, %stride) : !cute.layout<"(32,1):(1,128)">
    %coalesce = cute.coalesce(%lay_10) : (!cute.layout<"(32,1):(1,128)">) -> !cute.layout<"32:1">
    %coord = cute.make_coord() : () -> !cute.coord<"((128,16),1,4,6)">
    %coalesce_11 = cute.coalesce(%1, %coord) : (!cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,(1,6)):(((1,4096),64),0,1024,(0,8192))">, !cute.coord<"((128,16),1,4,6)">) -> !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
    %shape_12 = cute.make_shape() : () -> !cute.shape<"(64,8)">
    %stride_13 = cute.make_stride() : () -> !cute.stride<"(1,64)">
    %lay_14 = cute.make_layout(%shape_12, %stride_13) : !cute.layout<"(64,8):(1,64)">
    %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"0">
    %lay_15 = cute.make_composed_layout(%2, %int_tuple, %lay_14) : !cute.composed_layout<"S<3,4,3> o 0 o (64,8):(1,64)">
    %shape_16 = cute.make_shape() : () -> !cute.shape<"(128,32,4)">
    %int_tuple_17 = cute.make_int_tuple() : () -> !cute.int_tuple<"(1,0,2)">
    %tile_to_shape = cute.tile_to_shape(%lay_15, %shape_16, %int_tuple_17) : (!cute.composed_layout<"S<3,4,3> o 0 o (64,8):(1,64)">, !cute.shape<"(128,32,4)">, !cute.int_tuple<"(1,0,2)">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">
    %coord_18 = cute.make_coord() : () -> !cute.coord<"(_,_,_,0)">
    %slice = cute.slice(%coalesce_11, %coord_18) : !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, !cute.coord<"(_,_,_,0)">
    %7 = cute.get(%0) <{mode = [1]}> : !cute.layout<"((1,(1,1)),((128,16),(1,4))):((1@1,(0,0)),((1@0,1@1),(0,16@1)))"> -> !cute.layout<"((128,16),(1,4)):((1@0,1@1),(0,16@1))">
    %dice = cute.dice(%7, "(1,(1,1))") : (!cute.layout<"((128,16),(1,4)):((1@0,1@1),(0,16@1))">) -> !cute.layout<"((128,16),1,4):((1@0,1@1),0,16@1)">
    %non_exec_atom, %tma_tensor = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view, %slice, %dice) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_f16_2, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4):(((1,4096),64),0,1024)">, !cute.layout<"((128,16),1,4):((1@0,1@1),0,16@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">)
    %non_exec_atom_19, %tma_tensor_20 = cute_nvgpu.atom.make_non_exec_tiled_tma_load(%view_3, %slice, %dice) <{kind = <sm_90> num_multicast = 1}> : (!memref_gmem_f16_2, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4):(((1,4096),64),0,1024)">, !cute.layout<"((128,16),1,4):((1@0,1@1),0,16@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">)
    %8 = cute.select<[0, 1]> (%tile_to_shape) : (!cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">) -> !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4)):((1,2048),(64,512))">
    %9 = cute.get_shape(%5) : (!cute.layout<"(?,?,?):(1,?{i64},?{i64})">) -> !cute.shape<"(?,?,?)">
    %e0, %e1, %e2 = cute.get_leaves(%9) : !cute.shape<"(?,?,?)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_21 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %itup_22 = cute.to_int_tuple(%e2) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %shape_23 = cute.make_shape(%itup, %itup_21, %itup_22) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %10 = cute.make_identity_layout(%shape_23) : !cute.layout<"(?,?,?):(1@0,1@1,1@2)">
    %tile_24 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
    %11 = cute.composition(%10, %tile_24) : (!cute.layout<"(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;32:1]">) -> !cute.layout<"(128,32):(1@0,1@1)">
    %non_exec_atom_25, %tma_tensor_26 = cute_nvgpu.atom.make_non_exec_tiled_tma_store(%view_5, %8, %11) : (!memref_gmem_f16_2, !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4)):((1,2048),(64,512))">, !cute.layout<"(128,32):(1@0,1@1)">) -> (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">)
    %tile_27 = cute.make_tile() : () -> !cute.tile<"[128:1;128:1]">
    %div_28 = cute.zipped_divide(%view_5, %tile_27) : !memref_gmem_f16_2, !cute.tile<"[128:1;128:1]">
    %coord_29 = cute.make_coord() : () -> !cute.coord<"(0,(_,_,_))">
    %slice_30 = cute.slice(%div_28, %coord_29) : !memref_gmem_f16_3, !cute.coord<"(0,(_,_,_))">
    %lay_31 = cute.get_layout(%slice_30) : !memref_gmem_f16_4
    %12 = cute.get_shape(%lay_31) : (!cute.layout<"(?,?,?):(128,?{i64 div=128},?{i64})">) -> !cute.shape<"(?,?,?)">
    %e0_32, %e1_33, %e2_34 = cute.get_leaves(%12) : !cute.shape<"(?,?,?)">
    %itup_35 = cute.to_int_tuple(%e0_32) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %13 = cute.get_scalars(%itup_35) : !cute.int_tuple<"?">
    %itup_36 = cute.to_int_tuple(%e1_33) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %14 = cute.get_scalars(%itup_36) : !cute.int_tuple<"?">
    %itup_37 = cute.to_int_tuple(%e2_34) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %15 = cute.get_scalars(%itup_37) : !cute.int_tuple<"?">
    %int_tuple_38 = cute.make_int_tuple(%itup_35, %itup_36, %itup_37) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?,?)">
    %tile_39 = cute.make_tile() : () -> !cute.tile<"[1:0;1:0]">
    %shp = cute.ceil_div(%int_tuple_38, %tile_39) : !cute.int_tuple<"(?,?,?)">, !cute.tile<"[1:0;1:0]">
    %e0_40, %e1_41, %e2_42 = cute.get_leaves(%shp) : !cute.int_tuple<"(?,?,?)">
    %shape_43 = cute.make_shape(%e0_40, %e1_41, %e2_42) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %lay_44 = cute.make_layout(%shape_43) : !cute.layout<"(?,?,?):(1,?,?)">
    %16 = cute.get_shape(%lay_44) : (!cute.layout<"(?,?,?):(1,?,?)">) -> !cute.shape<"(?,?,?)">
    %e0_45, %e1_46, %e2_47 = cute.get_leaves(%16) : !cute.shape<"(?,?,?)">
    %itup_48 = cute.to_int_tuple(%e0_45) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %17 = cute.get_scalars(%itup_48) : !cute.int_tuple<"?">
    %itup_49 = cute.to_int_tuple(%e1_46) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %18 = cute.get_scalars(%itup_49) : !cute.int_tuple<"?">
    %19 = cute.fast_divmod.create_divisor(%17) : i32 -> !cute.fast_divmod_divisor<32>
    %20 = cute.fast_divmod.create_divisor(%18) : i32 -> !cute.fast_divmod_divisor<32>
    %int_tuple_50 = cute.make_int_tuple(%itup_48) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %sz = cute.size(%int_tuple_50) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %e0_51 = cute.get_leaves(%sz) : !cute.int_tuple<"?">
    %int_tuple_52 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
    %mul = cute.tuple_mul(%e0_51, %int_tuple_52) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %int_tuple_53 = cute.make_int_tuple(%itup_49) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %sz_54 = cute.size(%int_tuple_53) : (!cute.int_tuple<"?">) -> !cute.int_tuple<"?">
    %e0_55 = cute.get_leaves(%sz_54) : !cute.int_tuple<"?">
    %mul_56 = cute.tuple_mul(%e0_55, %int_tuple_52) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %itup_57 = cute.to_int_tuple(%e2_47) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %int_tuple_58 = cute.make_int_tuple(%mul, %mul_56, %itup_57) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?,?,?)">
    %sz_59 = cute.size(%int_tuple_58) : (!cute.int_tuple<"(?,?,?)">) -> !cute.int_tuple<"?">
    %e0_60 = cute.get_leaves(%sz_59) : !cute.int_tuple<"?">
    %21 = cute.get_scalars(%e0_60) : !cute.int_tuple<"?">
    %22 = arith.minsi %21, %c148_i32 : i32
    %23 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c192_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %c229632_i64, gridDim = (%c1_i32, %c1_i32, %22), stream = %arg3) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%23] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cluster_dim[%23] (%c1_i32, %c1_i32, %c1_i32) : !cuda.launch_cfg<max_attrs = 17>, i32, i32, i32
    cuda.launch_cfg.cooperative[%23] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %24 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_infraswe_b200_static_replay_2PersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA11281612_0<%23> (%6, %non_exec_atom, %tma_tensor, %non_exec_atom_19, %tma_tensor_20, %non_exec_atom_25, %tma_tensor_26, %div, %coalesce_11, %coalesce_11, %tile_to_shape, %lay_8, %coalesce, %13, %14, %15, %19, %20) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!mma_f16_f16_f32_128x128x16, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.layout<"((1),1,1,1):((0),0,0,0)">, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">, !cute.layout<"128:1">, !cute.layout<"32:1">, i32, i32, i32, !cute.fast_divmod_divisor<32>, !cute.fast_divmod_divisor<32>) -> !cuda.result
    %25 = cuda.cast %24 : !cuda.result -> i32
    cuda.return_if_error %25 : i32
    return %c0_i32 : i32
  }
}


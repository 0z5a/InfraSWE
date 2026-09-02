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
!memref_smem_i128 = !cute.memref<i128, smem, align<32>, "1:0">
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
    cuda.kernel @kernel_cutlass_kernel_infraswe_b200_dynamic_replay_2PersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161_0(%arg0: !mma_f16_f16_f32_128x128x16, %arg1: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg2: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg3: !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg4: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg5: !cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, %arg6: !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, %arg7: !cute.layout<"((1),1,1,1):((0),0,0,0)">, %arg8: !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, %arg9: !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, %arg10: !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">, %arg11: !cute.layout<"128:1">, %arg12: !cute.layout<"32:1">, %arg13: i32, %arg14: i32, %arg15: i32) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 224, 1, 1>} {
      %c127_i32 = arith.constant 127 : i32
      %c256_i32 = arith.constant 256 : i32
      %c229632_i32 = arith.constant 229632 : i32
      %false = arith.constant false
      %c160_i32 = arith.constant 160 : i32
      %c2_i32 = arith.constant 2 : i32
      %c16_i32 = arith.constant 16 : i32
      %c6_i32 = arith.constant 6 : i32
      %c32768_i32 = arith.constant 32768 : i32
      %c10000000_i32 = arith.constant 10000000 : i32
      %true = arith.constant true
      %c196864_i32 = arith.constant 196864 : i32
      %c98560_i32 = arith.constant 98560 : i32
      %c-128_i32 = arith.constant -128 : i32
      %c128_i32 = arith.constant 128 : i32
      %c224_i32 = arith.constant 224 : i32
      %c4_i32 = arith.constant 4 : i32
      %c176_i32 = arith.constant 176 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c5_i32 = arith.constant 5 : i32
      %c32_i32 = arith.constant 32 : i32
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
      %13 = nvvm.read.ptx.sreg.cluster.ctarank : i32
      %14 = cute_nvgpu.arch.make_warp_uniform(%13) : i32
      %15 = arith.cmpi eq, %14, %c0_i32 : i32
      %smem_ptr = cute_nvgpu.arch.get_dyn_smem() : !cute.ptr<i8, smem, align<1024>>
      %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"176">
      %ptr = cute.add_offset(%smem_ptr, %int_tuple) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"176">) -> !cute.ptr<i8, smem, align<16>>
      %smem_size = cute_nvgpu.arch.get_dyn_smem_size() : i32
      %16 = arith.cmpi sge, %smem_size, %c176_i32 : i32
      cf.assert %16, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 176 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %int_tuple_0 = cute.make_int_tuple() : () -> !cute.int_tuple<"96">
      %ptr_1 = cute.add_offset(%smem_ptr, %int_tuple_0) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"96">) -> !cute.ptr<i8, smem, align<32>>
      %int_tuple_2 = cute.make_int_tuple() : () -> !cute.int_tuple<"136">
      %ptr_3 = cute.add_offset(%smem_ptr, %int_tuple_2) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"136">) -> !cute.ptr<i8, smem, align<8>>
      %iter = cute.recast_iter(%ptr_3) : !cute.ptr<i8, smem, align<8>> to !cute.ptr<i32, smem, align<8>>
      %int_tuple_4 = cute.make_int_tuple() : () -> !cute.int_tuple<"144">
      %ptr_5 = cute.add_offset(%smem_ptr, %int_tuple_4) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"144">) -> !cute.ptr<i8, smem, align<16>>
      %int_tuple_6 = cute.make_int_tuple() : () -> !cute.int_tuple<"160">
      %ptr_7 = cute.add_offset(%smem_ptr, %int_tuple_6) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"160">) -> !cute.ptr<i8, smem, align<32>>
      %iter_8 = cute.recast_iter(%smem_ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<i64, smem, align<1024>>
      %17 = arith.cmpi eq, %11, %c0_i32 : i32
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %iter_8 : !cute.ptr<i64, smem, align<1024>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_37 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
        %ptr_38 = cute.add_offset(%iter_8, %int_tuple_37) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
        %43 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_39 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
        %ptr_40 = cute.add_offset(%iter_8, %int_tuple_39) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem, align<16>>
        %44 = builtin.unrealized_conversion_cast %ptr_40 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %44, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_41 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
        %ptr_42 = cute.add_offset(%iter_8, %int_tuple_41) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
        %45 = builtin.unrealized_conversion_cast %ptr_42 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %45, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_43 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
        %ptr_44 = cute.add_offset(%iter_8, %int_tuple_43) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"4">) -> !cute.ptr<i64, smem, align<32>>
        %46 = builtin.unrealized_conversion_cast %ptr_44 : !cute.ptr<i64, smem, align<32>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %46, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_45 = cute.make_int_tuple() : () -> !cute.int_tuple<"5">
        %ptr_46 = cute.add_offset(%iter_8, %int_tuple_45) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"5">) -> !cute.ptr<i64, smem>
        %47 = builtin.unrealized_conversion_cast %ptr_46 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %47, %c1_i32 : !llvm.ptr<3>, i32
      }
      %int_tuple_9 = cute.make_int_tuple() : () -> !cute.int_tuple<"6">
      %ptr_10 = cute.add_offset(%iter_8, %int_tuple_9) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"6">) -> !cute.ptr<i64, smem, align<16>>
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %ptr_10 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_37 = cute.make_int_tuple() : () -> !cute.int_tuple<"7">
        %ptr_38 = cute.add_offset(%iter_8, %int_tuple_37) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"7">) -> !cute.ptr<i64, smem>
        %43 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_39 = cute.make_int_tuple() : () -> !cute.int_tuple<"8">
        %ptr_40 = cute.add_offset(%iter_8, %int_tuple_39) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"8">) -> !cute.ptr<i64, smem, align<64>>
        %dyn = cute.derefine(%ptr_40) : !cute.ptr<i64, smem, align<64>> to !cute.ptr<i64, smem, align<16>>
        %44 = builtin.unrealized_conversion_cast %dyn : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %44, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_41 = cute.make_int_tuple() : () -> !cute.int_tuple<"9">
        %ptr_42 = cute.add_offset(%iter_8, %int_tuple_41) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"9">) -> !cute.ptr<i64, smem>
        %45 = builtin.unrealized_conversion_cast %ptr_42 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %45, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_43 = cute.make_int_tuple() : () -> !cute.int_tuple<"10">
        %ptr_44 = cute.add_offset(%iter_8, %int_tuple_43) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"10">) -> !cute.ptr<i64, smem, align<16>>
        %46 = builtin.unrealized_conversion_cast %ptr_44 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %46, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_45 = cute.make_int_tuple() : () -> !cute.int_tuple<"11">
        %ptr_46 = cute.add_offset(%iter_8, %int_tuple_45) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"11">) -> !cute.ptr<i64, smem>
        %47 = builtin.unrealized_conversion_cast %ptr_46 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %47, %c1_i32 : !llvm.ptr<3>, i32
      }
      %iter_11 = cute.recast_iter(%ptr_1) : !cute.ptr<i8, smem, align<32>> to !cute.ptr<i64, smem, align<32>>
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %iter_11 : !cute.ptr<i64, smem, align<32>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
        %int_tuple_37 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
        %ptr_38 = cute.add_offset(%iter_11, %int_tuple_37) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
        %43 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c1_i32 : !llvm.ptr<3>, i32
      }
      %int_tuple_12 = cute.make_int_tuple() : () -> !cute.int_tuple<"2">
      %ptr_13 = cute.add_offset(%iter_11, %int_tuple_12) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"2">) -> !cute.ptr<i64, smem, align<16>>
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %ptr_13 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c4_i32 : !llvm.ptr<3>, i32
        %int_tuple_37 = cute.make_int_tuple() : () -> !cute.int_tuple<"3">
        %ptr_38 = cute.add_offset(%iter_11, %int_tuple_37) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"3">) -> !cute.ptr<i64, smem>
        %43 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %43, %c4_i32 : !llvm.ptr<3>, i32
      }
      %iter_14 = cute.recast_iter(%ptr_5) : !cute.ptr<i8, smem, align<16>> to !cute.ptr<i64, smem, align<16>>
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %iter_14 : !cute.ptr<i64, smem, align<16>> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c1_i32 : !llvm.ptr<3>, i32
      }
      %int_tuple_15 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
      %ptr_16 = cute.add_offset(%iter_14, %int_tuple_15) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"1">) -> !cute.ptr<i64, smem>
      scf.if %17 {
        %42 = builtin.unrealized_conversion_cast %ptr_16 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.init.shared %42, %c224_i32 : !llvm.ptr<3>, i32
      }
      %18 = arith.remsi %0, %c32_i32 : i32
      %19 = arith.cmpi slt, %18, %c1_i32 : i32
      nvvm.fence.mbarrier.init
      %iter_17 = cute.recast_iter(%ptr_7) : !cute.ptr<i8, smem, align<32>> to !cute.ptr<i32, smem, align<32>>
      %20 = cute.composed_get_outer(%arg8) : (!cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">) -> !cute.layout<"(((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
      %21 = cute.ptrtoint(%ptr) : !cute.ptr<i8, smem, align<16>> to i32
      %22 = arith.addi %21, %c127_i32 : i32
      %23 = arith.andi %22, %c-128_i32 : i32
      %24 = arith.extsi %23 : i32 to i64
      %iv = cute.assume(%24) : (i64) -> !cute.i64<divby 128>
      %25 = cute.inttoptr(%iv) : !cute.i64<divby 128> to !cute.ptr<i8, smem, align<128>>
      %int_tuple_18 = cute.make_int_tuple() : () -> !cute.int_tuple<"98304">
      %ptr_19 = cute.add_offset(%25, %int_tuple_18) : (!cute.ptr<i8, smem, align<128>>, !cute.int_tuple<"98304">) -> !cute.ptr<i8, smem, align<128>>
      %26 = arith.cmpi sge, %smem_size, %c98560_i32 : i32
      cf.assert %26, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 98560 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_20 = cute.recast_iter(%25) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view = cute.make_view(%iter_20, %20) : !memref_smem_f16
      %27 = cute.composed_get_outer(%arg9) : (!cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">) -> !cute.layout<"(((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">
      %int_tuple_21 = cute.make_int_tuple() : () -> !cute.int_tuple<"196608">
      %ptr_22 = cute.add_offset(%25, %int_tuple_21) : (!cute.ptr<i8, smem, align<128>>, !cute.int_tuple<"196608">) -> !cute.ptr<i8, smem, align<128>>
      %28 = arith.cmpi sge, %smem_size, %c196864_i32 : i32
      cf.assert %28, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 196864 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_23 = cute.recast_iter(%ptr_19) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view_24 = cute.make_view(%iter_23, %27) : !memref_smem_f16
      %tile = cute.make_tile() : () -> !cute.tile<"[128:1;64:1]">
      %coord = cute.make_coord() : () -> !cute.coord<"(_,_,_)">
      %tiled_view = cute.local_tile(%arg2, %tile, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">
      %tiled_view_25 = cute.local_tile(%arg4, %tile, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;64:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">
      %tile_26 = cute.make_tile() : () -> !cute.tile<"[128:1;128:1]">
      %tiled_view_27 = cute.local_tile(%arg6, %tile_26, %coord) : (!cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[128:1;128:1]">, !cute.coord<"(_,_,_)">) -> !cute.coord_tensor<"(0,0,0)", "(128,128,?,?,?):(1@0,1@1,128@0,128@1,1@2)">
      %sz = cute.size(%tiled_view) <{mode = [3]}> : (!cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">) -> !cute.int_tuple<"?">
      %e0 = cute.get_leaves(%sz) : !cute.int_tuple<"?">
      %29 = cute.get_scalars(%e0) : !cute.int_tuple<"?">
      %coord_28 = cute.make_coord() : () -> !cute.coord<"0">
      %ptn_A = cute.tiled.mma.partition A (%arg0, %tiled_view, %coord_28) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">
      %ptn_B = cute.tiled.mma.partition B (%arg0, %tiled_view_25, %coord_28) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,64,?,?,?):(1@0,1@1,128@0,64@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">
      %ptn_C = cute.tiled.mma.partition C (%arg0, %tiled_view_27, %coord_28) : (!mma_f16_f16_f32_128x128x16, !cute.coord_tensor<"(0,0,0)", "(128,128,?,?,?):(1@0,1@1,128@0,128@1,1@2)">, !cute.coord<"0">) -> !cute.coord_tensor<"(0,0,0)", "((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">
      %shape = cute.make_shape() : () -> !cute.shape<"(1)">
      %lay = cute.make_layout(%shape) : !cute.layout<"(1):(0)">
      %grouped = cute.group_modes(%view) <0, 3> : (!memref_smem_f16) -> !memref_smem_f16_1
      %grouped_29 = cute.group_modes(%ptn_A) <0, 3> : (!cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">
      %res_smem_tensor, %res_target_tensors = cute_nvgpu.atom.tma_partition(%arg1, %coord_28, %lay, %grouped, %grouped_29) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"(1):(0)">, !memref_smem_f16_1, !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">) -> (!memref_smem_f16_2, !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">)
      %grouped_30 = cute.group_modes(%view_24) <0, 3> : (!memref_smem_f16) -> !memref_smem_f16_1
      %grouped_31 = cute.group_modes(%ptn_B) <0, 3> : (!cute.coord_tensor<"(0,0,0)", "((128,16),1,4,?,?,?):((1@0,1@1),0,16@1,128@0,64@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">
      %res_smem_tensor_32, %res_target_tensors_33 = cute_nvgpu.atom.tma_partition(%arg3, %coord_28, %lay, %grouped_30, %grouped_31) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"(1):(0)">, !memref_smem_f16_1, !cute.coord_tensor<"(0,0,0)", "(((128,16),1,4),?,?,?):(((1@0,1@1),0,16@1),128@0,64@1,1@2)">) -> (!memref_smem_f16_2, !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">)
      %frg_A = cute.mma.make_fragment A (%arg0, %view) : (!mma_f16_f16_f32_128x128x16, !memref_smem_f16) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">
      %frg_B = cute.mma.make_fragment B (%arg0, %view_24) : (!mma_f16_f16_f32_128x128x16, !memref_smem_f16) -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">
      %shape_34 = cute.make_shape() : () -> !cute.shape<"((128,128),1,1,2)">
      %frg_C = cute.mma.make_fragment C (%arg0, %shape_34) : (!mma_f16_f16_f32_128x128x16, !cute.shape<"((128,128),1,1,2)">) -> !memref_tmem_f32
      nvvm.barrier
      %30 = nvvm.read.ptx.sreg.ctaid.x : i32
      %31 = nvvm.read.ptx.sreg.ctaid.y : i32
      %32 = nvvm.read.ptx.sreg.ctaid.z : i32
      %33:7 = scf.if %12 -> (i32, i32, i32, i1, i32, i32, i32) {
        %42:9 = scf.while (%arg16 = %30, %arg17 = %31, %arg18 = %32, %arg19 = %true, %arg20 = %c0_i32, %arg21 = %c1_i32, %arg22 = %c0_i32, %arg23 = %c0_i32, %arg24 = %c0_i32) : (i32, i32, i32, i1, i32, i32, i32, i32, i32) -> (i32, i32, i32, i1, i32, i32, i32, i32, i32) {
          scf.condition(%arg19) %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24 : i32, i32, i32, i1, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg16: i32, %arg17: i32, %arg18: i32, %arg19: i1, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32):
          %coord_39 = cute.make_coord(%arg16, %arg18) : (i32, i32) -> !cute.coord<"(_,?,_,?)">
          %slice = cute.slice(%res_target_tensors, %coord_39) : !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">, !cute.coord<"(_,?,_,?)">
          %coord_40 = cute.make_coord(%arg17, %arg18) : (i32, i32) -> !cute.coord<"(_,?,_,?)">
          %slice_41 = cute.slice(%res_target_tensors_33, %coord_40) : !cute.coord_tensor<"(0,0,0)", "(((64,64),2),?,?,?):(((1@0,1@1),64@0),128@0,64@1,1@2)">, !cute.coord<"(_,?,_,?)">
          %int_tuple_42 = cute.make_int_tuple(%arg20) : (i32) -> !cute.int_tuple<"?">
          %ptr_43 = cute.add_offset(%ptr_10, %int_tuple_42) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %65 = builtin.unrealized_conversion_cast %ptr_43 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %66 = nvvm.mbarrier.wait.parity %65, %arg21 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
          %67:4 = scf.for %arg25 = %c0_i32 to %29 step %c1_i32 iter_args(%arg26 = %66, %arg27 = %c0_i32, %arg28 = %arg20, %arg29 = %arg21) -> (i1, i32, i32, i32)  : i32 {
            %83 = arith.extui %arg26 : i1 to i32
            %84 = arith.cmpi eq, %83, %c0_i32 : i32
            scf.if %84 {
              %int_tuple_88 = cute.make_int_tuple(%arg28) : (i32) -> !cute.int_tuple<"?">
              %ptr_89 = cute.add_offset(%ptr_10, %int_tuple_88) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %97 = builtin.unrealized_conversion_cast %ptr_89 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity.shared %97, %arg29, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %85 = nvvm.elect.sync -> i1
            scf.if %85 {
              %int_tuple_88 = cute.make_int_tuple(%arg28) : (i32) -> !cute.int_tuple<"?">
              %ptr_89 = cute.add_offset(%iter_8, %int_tuple_88) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %97 = builtin.unrealized_conversion_cast %ptr_89 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.txn %97, %c32768_i32 {kind = #nvvm.mbar_txn_kind<arrive_expect_tx>} : !llvm.ptr<3>, i32
            }
            %86 = arith.addi %arg28, %c1_i32 : i32
            %87 = arith.addi %arg27, %c1_i32 : i32
            %88 = arith.cmpi eq, %86, %c6_i32 : i32
            %89 = arith.select %88, %c0_i32, %86 : i32
            %90 = scf.if %88 -> (i32) {
              %97 = arith.xori %arg29, %c1_i32 : i32
              scf.yield %97 : i32
            } else {
              scf.yield %arg29 : i32
            }
            %coord_51 = cute.make_coord(%arg27) : (i32) -> !cute.coord<"(_,?)">
            %slice_52 = cute.slice(%slice, %coord_51) : !cute.coord_tensor<"(?{div=128},0,?)", "(((64,64),2),?):(((1@0,1@1),64@0),64@1)">, !cute.coord<"(_,?)">
            %iter_53 = cute.get_iter(%slice_52) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %tup = cute.deref_arith_tuple_iter(%iter_53) : !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %e0_54, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %coord_55 = cute.make_coord(%arg28) : (i32) -> !cute.coord<"(_,?)">
            %slice_56 = cute.slice(%res_smem_tensor, %coord_55) : !memref_smem_f16_2, !cute.coord<"(_,?)">
            %iter_57 = cute.get_iter(%slice_56) : !memref_smem_f16_3
            %int_tuple_58 = cute.make_int_tuple(%arg28) : (i32) -> !cute.int_tuple<"?">
            %ptr_59 = cute.add_offset(%iter_8, %int_tuple_58) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %lay_60 = cute.get_layout(%slice_52) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %shape_61 = cute.make_shape() : () -> !cute.shape<"1">
            %lay_62 = cute.make_layout(%shape_61) : !cute.layout<"1:0">
            %append = cute.append_to_rank<2> (%lay_60, %lay_62) : !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
            %int_tuple_63 = cute.make_int_tuple(%e0_54, %e1, %e2) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_63) : (!cute.int_tuple<"(?{div=128},?{div=64},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %view_64 = cute.make_view(%int_tup_iter, %append) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">
            %grouped_65 = cute.group_modes(%view_64) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">
            %lay_66 = cute.get_layout(%slice_56) : !memref_smem_f16_3
            %append_67 = cute.append_to_rank<2> (%lay_66, %lay_62) : !cute.layout<"((4096,2)):((1,4096))">, !cute.layout<"1:0">
            %view_68 = cute.make_view(%iter_57, %append_67) : !memref_smem_f16_4
            %grouped_69 = cute.group_modes(%view_68) <1, 2> : (!memref_smem_f16_4) -> !memref_smem_f16_5
            %91 = cute_nvgpu.atom.make_exec_tma(%arg1) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>
            %92 = cute_nvgpu.atom.set_value<tma_bar>(%91, %ptr_59) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.ptr<i64, smem>)
            cute.copy(%92, %grouped_65, %grouped_69) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">, !memref_smem_f16_5)
            %slice_70 = cute.slice(%slice_41, %coord_51) : !cute.coord_tensor<"(?{div=128},0,?)", "(((64,64),2),?):(((1@0,1@1),64@0),64@1)">, !cute.coord<"(_,?)">
            %iter_71 = cute.get_iter(%slice_70) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %tup_72 = cute.deref_arith_tuple_iter(%iter_71) : !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %e0_73, %e1_74, %e2_75 = cute.get_leaves(%tup_72) : !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %slice_76 = cute.slice(%res_smem_tensor_32, %coord_55) : !memref_smem_f16_2, !cute.coord<"(_,?)">
            %iter_77 = cute.get_iter(%slice_76) : !memref_smem_f16_3
            %lay_78 = cute.get_layout(%slice_70) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2)):(((1@0,1@1),64@0))">
            %append_79 = cute.append_to_rank<2> (%lay_78, %lay_62) : !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
            %int_tuple_80 = cute.make_int_tuple(%e0_73, %e1_74, %e2_75) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=64}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=64},?)">
            %int_tup_iter_81 = cute.make_arith_tuple_iter(%int_tuple_80) : (!cute.int_tuple<"(?{div=128},?{div=64},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=64},?)">
            %view_82 = cute.make_view(%int_tup_iter_81, %append_79) : !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">
            %grouped_83 = cute.group_modes(%view_82) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">
            %lay_84 = cute.get_layout(%slice_76) : !memref_smem_f16_3
            %append_85 = cute.append_to_rank<2> (%lay_84, %lay_62) : !cute.layout<"((4096,2)):((1,4096))">, !cute.layout<"1:0">
            %view_86 = cute.make_view(%iter_77, %append_85) : !memref_smem_f16_4
            %grouped_87 = cute.group_modes(%view_86) <1, 2> : (!memref_smem_f16_4) -> !memref_smem_f16_5
            %93 = cute_nvgpu.atom.make_exec_tma(%arg3) : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>
            %94 = cute_nvgpu.atom.set_value<tma_bar>(%93, %ptr_59) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.ptr<i64, smem>)
            cute.copy(%94, %grouped_83, %grouped_87) : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 65536, mode = tiled, num_cta = 1, g_stride = <"()"> tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">>, !cute.coord_tensor<"(?{div=128},?{div=64},?)", "(((64,64),2),(1)):(((1@0,1@1),64@0),(0))">, !memref_smem_f16_5)
            %95 = arith.cmpi sgt, %29, %87 : i32
            %96 = scf.if %95 -> (i1) {
              %int_tuple_88 = cute.make_int_tuple(%89) : (i32) -> !cute.int_tuple<"?">
              %ptr_89 = cute.add_offset(%ptr_10, %int_tuple_88) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %97 = builtin.unrealized_conversion_cast %ptr_89 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %98 = nvvm.mbarrier.wait.parity %97, %90 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %98 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %96, %87, %89, %90 : i1, i32, i32, i32
          } {loop_annotation = #loop_annotation}
          %int_tuple_44 = cute.make_int_tuple(%arg23) : (i32) -> !cute.int_tuple<"?">
          %ptr_45 = cute.add_offset(%iter_14, %int_tuple_44) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %68 = builtin.unrealized_conversion_cast %ptr_45 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %68, %arg24, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %iter_46 = cute.recast_iter(%iter_17) : !cute.ptr<i32, smem, align<32>> to !cute.ptr<i128, smem, align<32>>
          %shape_47 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_48 = cute.make_layout(%shape_47) : !cute.layout<"1:0">
          %view_49 = cute.make_view(%iter_46, %lay_48) : !memref_smem_i128
          %69 = cute.memref.load_vec(%view_49) : (!memref_smem_i128) -> vector<1xi128>
          %70 = vector.extract %69[0] : i128 from vector<1xi128>
          %71 = nvvm.clusterlaunchcontrol.query_cancel.is_canceled %70 : i1
          %72 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x %70 : i32
          %73 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y %70 : i32
          %74 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z %70 : i32
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          %ptr_50 = cute.add_offset(%ptr_16, %int_tuple_44) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %75 = builtin.unrealized_conversion_cast %ptr_50 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %76 = nvvm.mapa %75, %c0_i32 : !llvm.ptr<3> -> !llvm.ptr<7>
          %77 = llvm.addrspacecast %76 : !llvm.ptr<7> to !llvm.ptr<3>
          nvvm.mbarrier.txn %77, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>, space = #nvvm.mbar_space<cluster>} : !llvm.ptr<3>, i32
          %78 = arith.addi %arg23, %c1_i32 : i32
          %79 = arith.addi %arg22, %c1_i32 : i32
          %80 = arith.cmpi eq, %78, %c1_i32 : i32
          %81 = arith.select %80, %c0_i32, %78 : i32
          %82 = scf.if %80 -> (i32) {
            %83 = arith.xori %arg24, %c1_i32 : i32
            scf.yield %83 : i32
          } else {
            scf.yield %arg24 : i32
          }
          scf.yield %72, %73, %74, %71, %67#2, %67#3, %79, %81, %82 : i32, i32, i32, i1, i32, i32, i32, i32, i32
        }
        %43 = arith.addi %42#4, %c1_i32 : i32
        %44 = arith.cmpi eq, %43, %c6_i32 : i32
        %45 = arith.select %44, %c0_i32, %43 : i32
        %46 = scf.if %44 -> (i32) {
          %65 = arith.xori %42#5, %c1_i32 : i32
          scf.yield %65 : i32
        } else {
          scf.yield %42#5 : i32
        }
        %47 = arith.addi %45, %c1_i32 : i32
        %48 = arith.cmpi eq, %47, %c6_i32 : i32
        %49 = arith.select %48, %c0_i32, %47 : i32
        %50 = scf.if %48 -> (i32) {
          %65 = arith.xori %46, %c1_i32 : i32
          scf.yield %65 : i32
        } else {
          scf.yield %46 : i32
        }
        %51 = arith.addi %49, %c1_i32 : i32
        %52 = arith.cmpi eq, %51, %c6_i32 : i32
        %53 = arith.select %52, %c0_i32, %51 : i32
        %54 = scf.if %52 -> (i32) {
          %65 = arith.xori %50, %c1_i32 : i32
          scf.yield %65 : i32
        } else {
          scf.yield %50 : i32
        }
        %55 = arith.addi %53, %c1_i32 : i32
        %56 = arith.cmpi eq, %55, %c6_i32 : i32
        %57 = arith.select %56, %c0_i32, %55 : i32
        %58 = scf.if %56 -> (i32) {
          %65 = arith.xori %54, %c1_i32 : i32
          scf.yield %65 : i32
        } else {
          scf.yield %54 : i32
        }
        %59 = arith.addi %57, %c1_i32 : i32
        %60 = arith.cmpi eq, %59, %c6_i32 : i32
        %61 = arith.select %60, %c0_i32, %59 : i32
        %62 = scf.if %60 -> (i32) {
          %65 = arith.xori %58, %c1_i32 : i32
          scf.yield %65 : i32
        } else {
          scf.yield %58 : i32
        }
        %int_tuple_37 = cute.make_int_tuple(%61) : (i32) -> !cute.int_tuple<"?">
        %ptr_38 = cute.add_offset(%ptr_10, %int_tuple_37) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
        %63 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
        nvvm.mbarrier.try_wait.parity.shared %63, %62, %c10000000_i32 : !llvm.ptr<3>, i32, i32
        %64 = nvvm.elect.sync -> i1
        scf.if %64 {
          %ptr_39 = cute.add_offset(%iter_8, %int_tuple_37) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %65 = builtin.unrealized_conversion_cast %ptr_39 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.txn %65, %c32768_i32 {kind = #nvvm.mbar_txn_kind<arrive_expect_tx>} : !llvm.ptr<3>, i32
        }
        scf.yield %42#0, %42#1, %42#2, %42#3, %42#6, %42#7, %42#8 : i32, i32, i32, i1, i32, i32, i32
      } else {
        scf.yield %30, %31, %32, %true, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i1, i32, i32, i32
      }
      %34 = arith.cmpi eq, %11, %c6_i32 : i32
      %35 = arith.andi %34, %15 : i1
      %36:8 = scf.if %35 -> (i32, i32, i32, i1, i32, i32, i32, i32) {
        %42:11 = scf.while (%arg16 = %33#0, %arg17 = %33#1, %arg18 = %33#2, %arg19 = %33#3, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %c1_i32, %arg24 = %33#4, %arg25 = %33#5, %arg26 = %33#6) : (i32, i32, i32, i1, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i1, i32, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg19) %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26 : i32, i32, i32, i1, i32, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg16: i32, %arg17: i32, %arg18: i32, %arg19: i1, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32):
          %int_tuple_37 = cute.make_int_tuple(%arg22) : (i32) -> !cute.int_tuple<"?">
          %ptr_38 = cute.add_offset(%ptr_16, %int_tuple_37) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %43 = builtin.unrealized_conversion_cast %ptr_38 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %43, %arg23, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          scf.if %19 {
            %ptr_47 = cute.add_offset(%iter_14, %int_tuple_37) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %66 = builtin.unrealized_conversion_cast %ptr_47 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %67 = nvvm.mapa %66, %18 : !llvm.ptr<3> -> !llvm.ptr<7>
            %68 = llvm.addrspacecast %67 : !llvm.ptr<7> to !llvm.ptr<3>
            nvvm.mbarrier.txn %68, %c16_i32 {kind = #nvvm.mbar_txn_kind<arrive_expect_tx>, space = #nvvm.mbar_space<cluster>} : !llvm.ptr<3>, i32
          }
          %ptr_39 = cute.add_offset(%iter_14, %int_tuple_37) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %44 = nvvm.elect.sync -> i1
          scf.if %44 {
            %66 = builtin.unrealized_conversion_cast %ptr_39 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            %67 = builtin.unrealized_conversion_cast %iter_17 : !cute.ptr<i32, smem, align<32>> to !llvm.ptr<3>
            nvvm.clusterlaunchcontrol.try_cancel.multicast %67, %66
          }
          %45 = arith.addi %arg20, %c1_i32 : i32
          %46 = arith.addi %arg22, %c1_i32 : i32
          %47 = arith.addi %arg21, %c1_i32 : i32
          %48 = arith.cmpi eq, %46, %c1_i32 : i32
          %49 = arith.select %48, %c0_i32, %46 : i32
          %50 = scf.if %48 -> (i32) {
            %66 = arith.xori %arg23, %c1_i32 : i32
            scf.yield %66 : i32
          } else {
            scf.yield %arg23 : i32
          }
          %int_tuple_40 = cute.make_int_tuple(%arg25) : (i32) -> !cute.int_tuple<"?">
          %ptr_41 = cute.add_offset(%iter_14, %int_tuple_40) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %51 = builtin.unrealized_conversion_cast %ptr_41 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %51, %arg26, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %iter_42 = cute.recast_iter(%iter_17) : !cute.ptr<i32, smem, align<32>> to !cute.ptr<i128, smem, align<32>>
          %shape_43 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_44 = cute.make_layout(%shape_43) : !cute.layout<"1:0">
          %view_45 = cute.make_view(%iter_42, %lay_44) : !memref_smem_i128
          %52 = cute.memref.load_vec(%view_45) : (!memref_smem_i128) -> vector<1xi128>
          %53 = vector.extract %52[0] : i128 from vector<1xi128>
          %54 = nvvm.clusterlaunchcontrol.query_cancel.is_canceled %53 : i1
          %55 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x %53 : i32
          %56 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y %53 : i32
          %57 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z %53 : i32
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          %ptr_46 = cute.add_offset(%ptr_16, %int_tuple_40) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %58 = builtin.unrealized_conversion_cast %ptr_46 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %59 = nvvm.mapa %58, %c0_i32 : !llvm.ptr<3> -> !llvm.ptr<7>
          %60 = llvm.addrspacecast %59 : !llvm.ptr<7> to !llvm.ptr<3>
          nvvm.mbarrier.txn %60, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>, space = #nvvm.mbar_space<cluster>} : !llvm.ptr<3>, i32
          %61 = arith.addi %arg25, %c1_i32 : i32
          %62 = arith.addi %arg24, %c1_i32 : i32
          %63 = arith.cmpi eq, %61, %c1_i32 : i32
          %64 = arith.select %63, %c0_i32, %61 : i32
          %65 = scf.if %63 -> (i32) {
            %66 = arith.xori %arg26, %c1_i32 : i32
            scf.yield %66 : i32
          } else {
            scf.yield %arg26 : i32
          }
          scf.yield %55, %56, %57, %54, %45, %47, %49, %50, %62, %64, %65 : i32, i32, i32, i1, i32, i32, i32, i32, i32, i32, i32
        }
        scf.yield %42#0, %42#1, %42#2, %42#3, %42#4, %42#8, %42#9, %42#10 : i32, i32, i32, i1, i32, i32, i32, i32
      } else {
        scf.yield %33#0, %33#1, %33#2, %33#3, %c0_i32, %33#4, %33#5, %33#6 : i32, i32, i32, i1, i32, i32, i32, i32
      }
      %37 = arith.cmpi eq, %11, %c4_i32 : i32
      %38:7 = scf.if %37 -> (i32, i32, i32, i1, i32, i32, i32) {
        nvvm.barrier id = %c2_i32 number_of_threads = %c160_i32
        %tmem_ptr = cute_nvgpu.arch.sm100.retrieve_tmem_ptr(%iter) : !cute.ptr<i32, smem, align<8>> -> !cute.ptr<f32, tmem, align<16>>
        %lay_37 = cute.get_layout(%frg_C) : !memref_tmem_f32
        %view_38 = cute.make_view(%tmem_ptr, %lay_37) : !memref_tmem_f32_1
        %42:13 = scf.while (%arg16 = %36#0, %arg17 = %36#1, %arg18 = %36#2, %arg19 = %36#3, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %arg0, %arg23 = %c0_i32, %arg24 = %c0_i32, %arg25 = %c1_i32, %arg26 = %36#5, %arg27 = %36#6, %arg28 = %36#7) : (i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg19) %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28 : i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg16: i32, %arg17: i32, %arg18: i32, %arg19: i1, %arg20: i32, %arg21: i32, %arg22: !mma_f16_f16_f32_128x128x16, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32):
          %coord_39 = cute.make_coord(%arg24) : (i32) -> !cute.coord<"(_,_,_,?)">
          %slice = cute.slice(%view_38, %coord_39) : !memref_tmem_f32_1, !cute.coord<"(_,_,_,?)">
          %int_tuple_40 = cute.make_int_tuple(%arg20) : (i32) -> !cute.int_tuple<"?">
          %ptr_41 = cute.add_offset(%iter_8, %int_tuple_40) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %45 = builtin.unrealized_conversion_cast %ptr_41 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %46 = nvvm.mbarrier.wait.parity %45, %arg21 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
          %int_tuple_42 = cute.make_int_tuple(%arg24) : (i32) -> !cute.int_tuple<"?">
          %ptr_43 = cute.add_offset(%ptr_13, %int_tuple_42) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %47 = builtin.unrealized_conversion_cast %ptr_43 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %47, %arg25, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %48 = cute_nvgpu.atom.set_value<accum_c>(%arg22, %false) : (!mma_f16_f16_f32_128x128x16, i1)
          %49:5 = scf.for %arg29 = %c0_i32 to %29 step %c1_i32 iter_args(%arg30 = %46, %arg31 = %c0_i32, %arg32 = %arg20, %arg33 = %arg21, %arg34 = %48) -> (i1, i32, i32, i32, !mma_f16_f16_f32_128x128x16)  : i32 {
            %71 = arith.extui %arg30 : i1 to i32
            %72 = arith.cmpi eq, %71, %c0_i32 : i32
            scf.if %72 {
              %int_tuple_51 = cute.make_int_tuple(%arg32) : (i32) -> !cute.int_tuple<"?">
              %ptr_52 = cute.add_offset(%iter_8, %int_tuple_51) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_52 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.mbarrier.try_wait.parity.shared %82, %arg33, %c10000000_i32 : !llvm.ptr<3>, i32, i32
            }
            %73 = arith.addi %arg32, %c1_i32 : i32
            %74 = arith.addi %arg31, %c1_i32 : i32
            %75 = arith.cmpi eq, %73, %c6_i32 : i32
            %76 = arith.select %75, %c0_i32, %73 : i32
            %77 = scf.if %75 -> (i32) {
              %82 = arith.xori %arg33, %c1_i32 : i32
              scf.yield %82 : i32
            } else {
              scf.yield %arg33 : i32
            }
            %78 = scf.for %arg35 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg36 = %arg34) -> (!mma_f16_f16_f32_128x128x16)  : i32 {
              %coord_51 = cute.make_coord(%arg35, %arg32) : (i32, i32) -> !cute.coord<"(_,_,?,?)">
              %slice_52 = cute.slice(%frg_A, %coord_51) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">, !cute.coord<"(_,_,?,?)">
              %slice_53 = cute.slice(%frg_B, %coord_51) : !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1,4,6):(0,0,128,1024)">, !cute.coord<"(_,_,?,?)">
              cute.gemm(%arg36, %slice, %slice_52, %slice_53, %slice) : (!mma_f16_f16_f32_128x128x16, !memref_tmem_f32_2, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">, !memref_tmem_f32_2)
              %82 = cute_nvgpu.atom.set_value<accum_c>(%arg36, %true) : (!mma_f16_f16_f32_128x128x16, i1)
              scf.yield %82 : !mma_f16_f16_f32_128x128x16
            } {loop_annotation = #loop_annotation1}
            %79 = nvvm.elect.sync -> i1
            scf.if %79 {
              %int_tuple_51 = cute.make_int_tuple(%arg32) : (i32) -> !cute.int_tuple<"?">
              %ptr_52 = cute.add_offset(%ptr_10, %int_tuple_51) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_52 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              nvvm.tcgen05.commit %82 : !llvm.ptr<3>
            }
            %80 = arith.cmpi sgt, %29, %74 : i32
            %81 = scf.if %80 -> (i1) {
              %int_tuple_51 = cute.make_int_tuple(%76) : (i32) -> !cute.int_tuple<"?">
              %ptr_52 = cute.add_offset(%iter_8, %int_tuple_51) : (!cute.ptr<i64, smem, align<1024>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
              %82 = builtin.unrealized_conversion_cast %ptr_52 : !cute.ptr<i64, smem> to !llvm.ptr<3>
              %83 = nvvm.mbarrier.wait.parity %82, %77 {kind = #nvvm.mbar_wait<try>} : !llvm.ptr<3>, i32 -> i1
              scf.yield %83 : i1
            } else {
              scf.yield %true : i1
            }
            scf.yield %81, %74, %76, %77, %78 : i1, i32, i32, i32, !mma_f16_f16_f32_128x128x16
          }
          %50 = nvvm.elect.sync -> i1
          scf.if %50 {
            %ptr_51 = cute.add_offset(%iter_11, %int_tuple_42) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %71 = builtin.unrealized_conversion_cast %ptr_51 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.tcgen05.commit %71 : !llvm.ptr<3>
          }
          %51 = arith.addi %arg24, %c1_i32 : i32
          %52 = arith.addi %arg23, %c1_i32 : i32
          %53 = arith.cmpi eq, %51, %c2_i32 : i32
          %54 = arith.select %53, %c0_i32, %51 : i32
          %55 = scf.if %53 -> (i32) {
            %71 = arith.xori %arg25, %c1_i32 : i32
            scf.yield %71 : i32
          } else {
            scf.yield %arg25 : i32
          }
          %int_tuple_44 = cute.make_int_tuple(%arg27) : (i32) -> !cute.int_tuple<"?">
          %ptr_45 = cute.add_offset(%iter_14, %int_tuple_44) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %56 = builtin.unrealized_conversion_cast %ptr_45 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %56, %arg28, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %iter_46 = cute.recast_iter(%iter_17) : !cute.ptr<i32, smem, align<32>> to !cute.ptr<i128, smem, align<32>>
          %shape_47 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_48 = cute.make_layout(%shape_47) : !cute.layout<"1:0">
          %view_49 = cute.make_view(%iter_46, %lay_48) : !memref_smem_i128
          %57 = cute.memref.load_vec(%view_49) : (!memref_smem_i128) -> vector<1xi128>
          %58 = vector.extract %57[0] : i128 from vector<1xi128>
          %59 = nvvm.clusterlaunchcontrol.query_cancel.is_canceled %58 : i1
          %60 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x %58 : i32
          %61 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y %58 : i32
          %62 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z %58 : i32
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          %ptr_50 = cute.add_offset(%ptr_16, %int_tuple_44) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %63 = builtin.unrealized_conversion_cast %ptr_50 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %64 = nvvm.mapa %63, %c0_i32 : !llvm.ptr<3> -> !llvm.ptr<7>
          %65 = llvm.addrspacecast %64 : !llvm.ptr<7> to !llvm.ptr<3>
          nvvm.mbarrier.txn %65, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>, space = #nvvm.mbar_space<cluster>} : !llvm.ptr<3>, i32
          %66 = arith.addi %arg27, %c1_i32 : i32
          %67 = arith.addi %arg26, %c1_i32 : i32
          %68 = arith.cmpi eq, %66, %c1_i32 : i32
          %69 = arith.select %68, %c0_i32, %66 : i32
          %70 = scf.if %68 -> (i32) {
            %71 = arith.xori %arg28, %c1_i32 : i32
            scf.yield %71 : i32
          } else {
            scf.yield %arg28 : i32
          }
          scf.yield %60, %61, %62, %59, %49#2, %49#3, %49#4, %52, %54, %55, %67, %69, %70 : i32, i32, i32, i1, i32, i32, !mma_f16_f16_f32_128x128x16, i32, i32, i32, i32, i32, i32
        }
        %43 = arith.remsi %14, %c2_i32 : i32
        %44 = arith.cmpi eq, %43, %c0_i32 : i32
        scf.if %44 {
          %45 = arith.addi %42#8, %c1_i32 : i32
          %46 = arith.cmpi eq, %45, %c2_i32 : i32
          %47 = arith.select %46, %c0_i32, %45 : i32
          %48 = scf.if %46 -> (i32) {
            %50 = arith.xori %42#9, %c1_i32 : i32
            scf.yield %50 : i32
          } else {
            scf.yield %42#9 : i32
          }
          %int_tuple_39 = cute.make_int_tuple(%47) : (i32) -> !cute.int_tuple<"?">
          %ptr_40 = cute.add_offset(%ptr_13, %int_tuple_39) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %49 = builtin.unrealized_conversion_cast %ptr_40 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %49, %48, %c10000000_i32 : !llvm.ptr<3>, i32, i32
        }
        scf.yield %42#0, %42#1, %42#2, %42#3, %42#10, %42#11, %42#12 : i32, i32, i32, i1, i32, i32, i32
      } else {
        scf.yield %36#0, %36#1, %36#2, %36#3, %36#5, %36#6, %36#7 : i32, i32, i32, i1, i32, i32, i32
      }
      %39 = cute.composed_get_outer(%arg10) : (!cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">) -> !cute.layout<"((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">
      %40 = arith.cmpi sge, %smem_size, %c229632_i32 : i32
      cf.assert %40, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 229632 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_35 = cute.recast_iter(%ptr_22) : !cute.ptr<i8, smem, align<128>> to !cute.ptr<f16, smem, align<128>, S<3,4,3>>
      %view_36 = cute.make_view(%iter_35, %39) : !memref_smem_f16_6
      %41 = arith.cmpi slt, %11, %c4_i32 : i32
      scf.if %41 {
        scf.if %17 {
          cute_nvgpu.arch.sm100.alloc_tmem(%c256_i32, %iter) [ cta_1] : i32, !cute.ptr<i32, smem, align<8>>
        }
        nvvm.barrier id = %c2_i32 number_of_threads = %c160_i32
        %tmem_ptr = cute_nvgpu.arch.sm100.retrieve_tmem_ptr(%iter) : !cute.ptr<i32, smem, align<8>> -> !cute.ptr<f32, tmem, align<16>>
        %42:9 = scf.while (%arg16 = %c0_i32, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %38#0, %arg20 = %38#1, %arg21 = %38#2, %arg22 = %38#3, %arg23 = %38#4, %arg24 = %38#5, %arg25 = %38#6) : (i32, i32, i32, i32, i32, i32, i1, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
          scf.condition(%arg22) %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg23, %arg24, %arg25 : i32, i32, i32, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32):
          %lay_37 = cute.get_layout(%ptn_C) : !cute.coord_tensor<"(0,0,0)", "((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">
          %43 = cute.get_shape(%lay_37) : (!cute.layout<"((128,128),1,1,?,?,?):((1@0,1@1),0,0,128@0,128@1,1@2)">) -> !cute.shape<"((128,128),1,1,?,?,?)">
          %e0_38, %e1, %e2, %e3, %e4, %e5, %e6 = cute.get_leaves(%43) : !cute.shape<"((128,128),1,1,?,?,?)">
          %itup = cute.to_int_tuple(%e4) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %itup_39 = cute.to_int_tuple(%e5) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %itup_40 = cute.to_int_tuple(%e6) : !cute.shape<"?"> to !cute.int_tuple<"?">
          %shape_41 = cute.make_shape(%itup, %itup_39, %itup_40) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"((128,1),(128,1),?,?,?)">
          %stride = cute.make_stride() : () -> !cute.stride<"((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %lay_42 = cute.make_layout(%shape_41, %stride) : !cute.layout<"((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %int_tuple_43 = cute.make_int_tuple() : () -> !cute.int_tuple<"(0,0,0)">
          %int_tup_iter = cute.make_arith_tuple_iter(%int_tuple_43) : (!cute.int_tuple<"(0,0,0)">) -> !cute.arith_tuple_iter<"(0,0,0)">
          %view_44 = cute.make_view(%int_tup_iter, %lay_42) : !cute.coord_tensor<"(0,0,0)", "((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">
          %shape_45 = cute.make_shape() : () -> !cute.shape<"((128,1),(128,1),2)">
          %stride_46 = cute.make_stride() : () -> !cute.stride<"((65536,0),(1,0),128)">
          %lay_47 = cute.make_layout(%shape_45, %stride_46) : !cute.layout<"((128,1),(128,1),2):((65536,0),(1,0),128)">
          %view_48 = cute.make_view(%tmem_ptr, %lay_47) : !memref_tmem_f32_3
          %atom = cute.make_atom() : () -> !cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x4>
          %tile_49 = cute.make_tile() : () -> !cute.tile<"[128:1;32:1]">
          %div = cute.flat_divide(%view_48, %tile_49) : !memref_tmem_f32_3, !cute.tile<"[128:1;32:1]">
          %coord_50 = cute.make_coord() : () -> !cute.coord<"(_,_,0,0,0)">
          %slice = cute.slice(%div, %coord_50) : !memref_tmem_f32_4, !cute.coord<"(_,_,0,0,0)">
          %44 = cute_nvgpu.atom.make_tmem_copy(%atom, %slice) : (!cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x4>, !memref_tmem_f32_5) -> !copy_ldtm_256
          %coord_51 = cute.make_coord(%0) : (i32) -> !cute.coord<"?">
          %src_partitioned = cute.tiled.copy.partition_S(%44, %div, %coord_51) : (!copy_ldtm_256, !memref_tmem_f32_4, !cute.coord<"?">) -> !memref_tmem_f32_6
          %rmem = cute.memref.alloca() : !memref_rmem_f32
          %iter_52 = cute.get_iter(%rmem) : !memref_rmem_f32
          %rmem_53 = cute.memref.alloca() : !memref_rmem_f16
          %atom_54 = cute.make_atom() : () -> !cute_nvgpu.atom.stsm<f16, mode = <"(8,8)">, num_matrices = 4, t>
          %45 = cute.make_tiled_copy(%atom_54) : !copy_stsm_4
          %dst_partitioned = cute.tiled.copy.partition_D(%45, %view_36, %coord_51) : (!copy_stsm_4, !memref_smem_f16_6, !cute.coord<"?">) -> !memref_smem_f16_7
          %retiled = cute.tiled.copy.retile(%45, %rmem_53) : (!copy_stsm_4, !memref_rmem_f16) -> !memref_rmem_f16_1
          %div_55 = cute.flat_divide(%view_44, %tile_49) : !cute.coord_tensor<"(0,0,0)", "((128,1),(128,1),?,?,?):((1@0,0),(1@1,0),128@0,128@1,1@2)">, !cute.tile<"[128:1;32:1]">
          %shape_56 = cute.make_shape() : () -> !cute.shape<"1">
          %lay_57 = cute.make_layout(%shape_56) : !cute.layout<"1:0">
          %grouped_58 = cute.group_modes(%view_36) <0, 2> : (!memref_smem_f16_6) -> !memref_smem_f16_8
          %grouped_59 = cute.group_modes(%div_55) <0, 2> : (!cute.coord_tensor<"(0,0,0)", "(128,32,1,4,?,?,?):(1@0,1@1,0,32@1,128@0,128@1,1@2)">) -> !cute.coord_tensor<"(0,0,0)", "((128,32),1,4,?,?,?):((1@0,1@1),0,32@1,128@0,128@1,1@2)">
          %res_smem_tensor_60, %res_target_tensors_61 = cute_nvgpu.atom.tma_partition(%arg5, %coord_28, %lay_57, %grouped_58, %grouped_59) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord<"0">, !cute.layout<"1:0">, !memref_smem_f16_8, !cute.coord_tensor<"(0,0,0)", "((128,32),1,4,?,?,?):((1@0,1@1),0,32@1,128@0,128@1,1@2)">) -> (!memref_smem_f16_9, !cute.coord_tensor<"(0,0,0)", "(((64,32),2),1,4,?,?,?):(((1@0,1@1),64@0),0,32@1,128@0,128@1,1@2)">)
          %coord_62 = cute.make_coord(%arg19, %arg20, %arg21) : (i32, i32, i32) -> !cute.coord<"(_,_,_,?,?,?)">
          %slice_63 = cute.slice(%res_target_tensors_61, %coord_62) : !cute.coord_tensor<"(0,0,0)", "(((64,32),2),1,4,?,?,?):(((1@0,1@1),64@0),0,32@1,128@0,128@1,1@2)">, !cute.coord<"(_,_,_,?,?,?)">
          %coord_64 = cute.make_coord(%arg17) : (i32) -> !cute.coord<"(_,_,_,_,_,?)">
          %slice_65 = cute.slice(%src_partitioned, %coord_64) : !memref_tmem_f32_6, !cute.coord<"(_,_,_,_,_,?)">
          %int_tuple_66 = cute.make_int_tuple(%arg17) : (i32) -> !cute.int_tuple<"?">
          %ptr_67 = cute.add_offset(%iter_11, %int_tuple_66) : (!cute.ptr<i64, smem, align<32>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %46 = builtin.unrealized_conversion_cast %ptr_67 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %46, %arg18, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %grouped_68 = cute.group_modes(%slice_65) <3, 5> : (!memref_tmem_f32_7) -> !memref_tmem_f32_8
          %grouped_69 = cute.group_modes(%slice_63) <1, 3> : (!cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),1,4):(((1@0,1@1),64@0),0,32@1)">) -> !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),(1,4)):(((1@0,1@1),64@0),(0,32@1))">
          %47 = arith.muli %36#4, %c4_i32 : i32
          scf.for %arg25 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
            %iter_75 = cute.get_iter(%retiled) : !memref_rmem_f16_1
            %coord_76 = cute.make_coord(%arg25) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_77 = cute.slice(%grouped_68, %coord_76) : !memref_tmem_f32_8, !cute.coord<"(_,_,_,?)">
            %iter_78 = cute.get_iter(%slice_77) : !memref_tmem_f32_9
            %lay_79 = cute.get_layout(%slice_77) : !memref_tmem_f32_9
            %append = cute.append_to_rank<2> (%lay_79, %lay_57) : !cute.layout<"(((32,16),1),2,1):(((1,65536),0),1048576,0)">, !cute.layout<"1:0">
            %view_80 = cute.make_view(%iter_78, %append) : !memref_tmem_f32_9
            %grouped_81 = cute.group_modes(%view_80) <1, 3> : (!memref_tmem_f32_9) -> !memref_tmem_f32_10
            %lay_82 = cute.get_layout(%rmem) : !memref_rmem_f32
            %append_83 = cute.append_to_rank<2> (%lay_82, %lay_57) : !cute.layout<"(((2,2,4),1),2,1):(((1,2,4),0),16,0)">, !cute.layout<"1:0">
            %view_84 = cute.make_view(%iter_52, %append_83) : !memref_rmem_f32
            %grouped_85 = cute.group_modes(%view_84) <1, 3> : (!memref_rmem_f32) -> !memref_rmem_f32_1
            cute.copy(%44, %grouped_81, %grouped_85) : (!copy_ldtm_256, !memref_tmem_f32_10, !memref_rmem_f32_1)
            %retiled_86 = cute.tiled.copy.retile(%45, %rmem) : (!copy_stsm_4, !memref_rmem_f32) -> !memref_rmem_f32_2
            %69 = cute.memref.load_vec(%retiled_86) : (!memref_rmem_f32_2) -> vector<32xf32>
            %70 = arith.truncf %69 : vector<32xf32> to vector<32xf16>
            cute.memref.store_vec(%70, %retiled) : (vector<32xf16>, !memref_rmem_f16_1) -> ()
            %71 = arith.addi %47, %arg25 : i32
            %72 = arith.remsi %71, %c4_i32 : i32
            %coord_87 = cute.make_coord(%72) : (i32) -> !cute.coord<"(_,_,_,?)">
            %slice_88 = cute.slice(%dst_partitioned, %coord_87) : !memref_smem_f16_7, !cute.coord<"(_,_,_,?)">
            %iter_89 = cute.get_iter(%slice_88) : !memref_smem_f16_10
            %lay_90 = cute.get_layout(%retiled) : !memref_rmem_f16_1
            %append_91 = cute.append_to_rank<2> (%lay_90, %lay_57) : !cute.layout<"((8,2),2,1):((1,8),16,0)">, !cute.layout<"1:0">
            %view_92 = cute.make_view(%iter_75, %append_91) : !memref_rmem_f16_1
            %grouped_93 = cute.group_modes(%view_92) <1, 3> : (!memref_rmem_f16_1) -> !memref_rmem_f16_2
            %lay_94 = cute.get_layout(%slice_88) : !memref_smem_f16_10
            %append_95 = cute.append_to_rank<2> (%lay_94, %lay_57) : !cute.layout<"((8,2),2,1):((1,1024),16,0)">, !cute.layout<"1:0">
            %view_96 = cute.make_view(%iter_89, %append_95) : !memref_smem_f16_10
            %grouped_97 = cute.group_modes(%view_96) <1, 3> : (!memref_smem_f16_10) -> !memref_smem_f16_11
            cute.copy(%45, %grouped_93, %grouped_97) : (!copy_stsm_4, !memref_rmem_f16_2, !memref_smem_f16_11)
            nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
            nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
            scf.if %17 {
              %coord_98 = cute.make_coord(%72) : (i32) -> !cute.coord<"(_,?)">
              %slice_99 = cute.slice(%res_smem_tensor_60, %coord_98) : !memref_smem_f16_9, !cute.coord<"(_,?)">
              %iter_100 = cute.get_iter(%slice_99) : !memref_smem_f16_12
              %coord_101 = cute.make_coord(%arg25) : (i32) -> !cute.coord<"(_,?)">
              %slice_102 = cute.slice(%grouped_69, %coord_101) : !cute.coord_tensor<"(?{div=128},?{div=128},?)", "(((64,32),2),(1,4)):(((1@0,1@1),64@0),(0,32@1))">, !cute.coord<"(_,?)">
              %iter_103 = cute.get_iter(%slice_102) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2)):(((1@0,1@1),64@0))">
              %tup = cute.deref_arith_tuple_iter(%iter_103) : !cute.arith_tuple_iter<"(?{div=128},?{div=32},?)">
              %e0_104, %e1_105, %e2_106 = cute.get_leaves(%tup) : !cute.int_tuple<"(?{div=128},?{div=32},?)">
              %lay_107 = cute.get_layout(%slice_99) : !memref_smem_f16_12
              %append_108 = cute.append_to_rank<2> (%lay_107, %lay_57) : !cute.layout<"((2048,2)):((1,2048))">, !cute.layout<"1:0">
              %view_109 = cute.make_view(%iter_100, %append_108) : !memref_smem_f16_13
              %grouped_110 = cute.group_modes(%view_109) <1, 2> : (!memref_smem_f16_13) -> !memref_smem_f16_14
              %lay_111 = cute.get_layout(%slice_102) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2)):(((1@0,1@1),64@0))">
              %append_112 = cute.append_to_rank<2> (%lay_111, %lay_57) : !cute.layout<"(((64,32),2)):(((1@0,1@1),64@0))">, !cute.layout<"1:0">
              %int_tuple_113 = cute.make_int_tuple(%e0_104, %e1_105, %e2_106) : (!cute.int_tuple<"?{div=128}">, !cute.int_tuple<"?{div=32}">, !cute.int_tuple<"?">) -> !cute.int_tuple<"(?{div=128},?{div=32},?)">
              %int_tup_iter_114 = cute.make_arith_tuple_iter(%int_tuple_113) : (!cute.int_tuple<"(?{div=128},?{div=32},?)">) -> !cute.arith_tuple_iter<"(?{div=128},?{div=32},?)">
              %view_115 = cute.make_view(%int_tup_iter_114, %append_112) : !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),1):(((1@0,1@1),64@0),0)">
              %grouped_116 = cute.group_modes(%view_115) <1, 2> : (!cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),1):(((1@0,1@1),64@0),0)">) -> !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),(1)):(((1@0,1@1),64@0),(0))">
              %73 = cute_nvgpu.atom.make_exec_tma(%arg5) : (!cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>) -> !cute_nvgpu.atom.tma_store<f16, copy_bits = 32768, mode = tiled, g_stride = <"()"> tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">>
              cute.copy(%73, %grouped_110, %grouped_116) : (!cute_nvgpu.atom.tma_store<f16, copy_bits = 32768, mode = tiled, g_stride = <"()"> tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">>, !memref_smem_f16_14, !cute.coord_tensor<"(?{div=128},?{div=32},?)", "(((64,32),2),(1)):(((1@0,1@1),64@0),(0))">)
              nvvm.cp.async.bulk.commit.group
              nvvm.cp.async.bulk.wait_group 3 {read}
            }
            nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
          }
          nvvm.barrier id = %c1_i32 number_of_threads = %c128_i32
          %48 = nvvm.elect.sync -> i1
          scf.if %48 {
            %ptr_75 = cute.add_offset(%ptr_13, %int_tuple_66) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
            %69 = builtin.unrealized_conversion_cast %ptr_75 : !cute.ptr<i64, smem> to !llvm.ptr<3>
            nvvm.mbarrier.txn %69, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>} : !llvm.ptr<3>, i32
          }
          %49 = arith.addi %arg17, %c1_i32 : i32
          %50 = arith.addi %arg16, %c1_i32 : i32
          %51 = arith.cmpi eq, %49, %c2_i32 : i32
          %52 = arith.select %51, %c0_i32, %49 : i32
          %53 = scf.if %51 -> (i32) {
            %69 = arith.xori %arg18, %c1_i32 : i32
            scf.yield %69 : i32
          } else {
            scf.yield %arg18 : i32
          }
          %int_tuple_70 = cute.make_int_tuple(%arg23) : (i32) -> !cute.int_tuple<"?">
          %ptr_71 = cute.add_offset(%iter_14, %int_tuple_70) : (!cute.ptr<i64, smem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %54 = builtin.unrealized_conversion_cast %ptr_71 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          nvvm.mbarrier.try_wait.parity.shared %54, %arg24, %c10000000_i32 : !llvm.ptr<3>, i32, i32
          %iter_72 = cute.recast_iter(%iter_17) : !cute.ptr<i32, smem, align<32>> to !cute.ptr<i128, smem, align<32>>
          %view_73 = cute.make_view(%iter_72, %lay_57) : !memref_smem_i128
          %55 = cute.memref.load_vec(%view_73) : (!memref_smem_i128) -> vector<1xi128>
          %56 = vector.extract %55[0] : i128 from vector<1xi128>
          %57 = nvvm.clusterlaunchcontrol.query_cancel.is_canceled %56 : i1
          %58 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x %56 : i32
          %59 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y %56 : i32
          %60 = nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z %56 : i32
          nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
          %ptr_74 = cute.add_offset(%ptr_16, %int_tuple_70) : (!cute.ptr<i64, smem>, !cute.int_tuple<"?">) -> !cute.ptr<i64, smem>
          %61 = builtin.unrealized_conversion_cast %ptr_74 : !cute.ptr<i64, smem> to !llvm.ptr<3>
          %62 = nvvm.mapa %61, %c0_i32 : !llvm.ptr<3> -> !llvm.ptr<7>
          %63 = llvm.addrspacecast %62 : !llvm.ptr<7> to !llvm.ptr<3>
          nvvm.mbarrier.txn %63, %c1_i32 {kind = #nvvm.mbar_txn_kind<arrive>, space = #nvvm.mbar_space<cluster>} : !llvm.ptr<3>, i32
          %64 = arith.addi %arg23, %c1_i32 : i32
          %65 = arith.addi %arg22, %c1_i32 : i32
          %66 = arith.cmpi eq, %64, %c1_i32 : i32
          %67 = arith.select %66, %c0_i32, %64 : i32
          %68 = scf.if %66 -> (i32) {
            %69 = arith.xori %arg24, %c1_i32 : i32
            scf.yield %69 : i32
          } else {
            scf.yield %arg24 : i32
          }
          scf.yield %50, %52, %53, %58, %59, %60, %57, %65, %67, %68 : i32, i32, i32, i32, i32, i32, i1, i32, i32, i32
        }
        nvvm.cp.async.bulk.wait_group 0 {read}
        scf.if %17 {
          cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit [ cta_1]
        }
        scf.if %17 {
          cute_nvgpu.arch.sm100.dealloc_tmem(%tmem_ptr, %c256_i32) [ cta_1] : !cute.ptr<f32, tmem, align<16>>, i32
        }
      }
      return
    }
  }
  func.func @cutlass_bmm_infraswe_b200_dynamic_replay_2PersistentDenseGemmKernelobjectat_Tensorgmemoi641i64_Tensorgmemoi64i641_Tensorgmemoi641i64_148_FakeStream_functionrunlocalslambdaat(%arg0: !memref_gmem_f16, %arg1: !memref_gmem_f16_1, %arg2: !memref_gmem_f16, %arg3: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c229632_i64 = arith.constant 229632 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c224_i32 = arith.constant 224 : i32
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
    %int_tuple_38 = cute.make_int_tuple() : () -> !cute.int_tuple<"1">
    %add = cute.tuple_add(%itup_35, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %sub = cute.tuple_sub(%add, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div_39 = cute.tuple_div(%sub, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %mul = cute.tuple_mul(%div_39, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %16 = cute.get_scalars(%mul) : !cute.int_tuple<"?">
    %add_40 = cute.tuple_add(%itup_36, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %sub_41 = cute.tuple_sub(%add_40, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div_42 = cute.tuple_div(%sub_41, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %mul_43 = cute.tuple_mul(%div_42, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %17 = cute.get_scalars(%mul_43) : !cute.int_tuple<"?">
    %add_44 = cute.tuple_add(%itup_37, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %sub_45 = cute.tuple_sub(%add_44, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %div_46 = cute.tuple_div(%sub_45, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %mul_47 = cute.tuple_mul(%div_46, %int_tuple_38) : (!cute.int_tuple<"?">, !cute.int_tuple<"1">) -> !cute.int_tuple<"?">
    %18 = cute.get_scalars(%mul_47) : !cute.int_tuple<"?">
    %19 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c224_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %c229632_i64, gridDim = (%16, %17, %18), stream = %arg3) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%19] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cluster_dim[%19] (%c1_i32, %c1_i32, %c1_i32) : !cuda.launch_cfg<max_attrs = 17>, i32, i32, i32
    cuda.launch_cfg.cooperative[%19] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %20 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_infraswe_b200_dynamic_replay_2PersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161_0<%19> (%6, %non_exec_atom, %tma_tensor, %non_exec_atom_19, %tma_tensor_20, %non_exec_atom_25, %tma_tensor_26, %div, %coalesce_11, %coalesce_11, %tile_to_shape, %lay_8, %coalesce, %13, %14, %15) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!mma_f16_f16_f32_128x128x16, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 65536, tma_gbasis = <"(64,64,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute_nvgpu.atom.non_exec_tiled_tma_store<f16, copy_bits = 32768, tma_gbasis = <"(64,32,1):(1@0,1@1,1@2)">, tma_format = F16_RN>, !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.layout<"((1),1,1,1):((0),0,0,0)">, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, !cute.composed_layout<"S<3,4,3> o 0 o (((64,2),16),1,4,6):(((1,4096),64),0,1024,8192)">, !cute.composed_layout<"S<3,4,3> o 0 o ((64,2),(8,4),(1,4)):((1,2048),(64,512),(0,4096))">, !cute.layout<"128:1">, !cute.layout<"32:1">, i32, i32, i32) -> !cuda.result
    %21 = cuda.cast %20 : !cuda.result -> i32
    cuda.return_if_error %21 : i32
    return %c0_i32 : i32
  }
}


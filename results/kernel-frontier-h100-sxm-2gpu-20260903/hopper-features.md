# Hopper SM90 架构新特性补测：TMA、WGMMA 与 multimem/NVLS

GPU：NVIDIA H100 80GB HBM3；CC 9.0；SM 132；显存 79.2 GiB。

## 结果

| 特性 | 运行状态 | 编译/指令门禁 | 正确性 | 说明 |
|---|---|---|---|---|
| TMA | PASS | PASS | PASS | 3 个 fresh-process replay；Triton tensor descriptor 实际执行 |
| WGMMA | PASS | PASS | PASS | 3 个 fresh-process replay；Triton `tl.dot` 实际执行 |
| multimem | passed | PASS | PASS | 合法 CUDA multicast 映射上的双卡硬件归约 |

## TMA 实测

- Driver `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED`：`1`。
- 4096×4096 BF16 copy+add：TMA candidate 中位数 `30.350 µs`；Torch add-out reference `25.565 µs`；比值 `1.187×`。
- 3 replay 均通过逐元素正确性、动态输入变化、CUDA Profiler 和编译产物指令门禁。
- SASS 明确包含 `UTMALDG.2D` 与 `UTMASTG.2D`。

## WGMMA 实测

- 1024³ FP16 GEMM：WGMMA candidate 中位数 `18.206 µs`；Torch MM reference `12.661 µs`；比值 `1.438×`。
- 3 replay 均通过正确性、CUDA Profiler 与编译产物指令门禁。
- PTX/SASS 明确包含 `wgmma.mma_async` / `HGMMA`。

## multimem/NVLS 实测

- Driver `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`：`1`；fabric handle：`1`；可见 GPU：`2`。
- 状态：`passed`；运行正确性：`True`。
- 每个 replay 都为两张 GPU 创建独立物理 backing、加入同一 multicast team，再从两张卡分别执行 `multimem.ld_reduce` 并校验归约结果。
- 编译后的 SM90 SASS 明确包含 `LDGMC.E.ADD.32.STRONG.SYS`。
- 普通指针不是 multimem address；本测试只在 Driver API 创建并绑定的合法 multicast 地址上执行。

## 证据边界

- TMA、WGMMA 与 multimem/NVLS 均为本机真实执行与指令级证据。
- multimem 结论仅适用于本次两卡可见的 NVSwitch multicast team；不外推到无 multicast 能力位的 PCIe/单卡拓扑。
- 原始 replay、Triton PTX/cubin/SASS、multimem PTX/cubin/SASS/二进制、Driver 属性与拓扑快照全部随 ZIP 提供。

参考：NVIDIA PTX ISA 的 `multimem.*` 定义与 CUDA Driver API multicast 管理。

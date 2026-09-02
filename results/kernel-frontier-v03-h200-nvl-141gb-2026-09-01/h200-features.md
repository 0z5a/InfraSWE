# H200 SM90 架构新特性补测：TMA 与 multimem

GPU：NVIDIA H200 NVL；CC 9.0；SM 132；显存 139.8 GiB。

## 结果

| 特性 | 运行状态 | 编译/指令门禁 | 正确性 | 说明 |
|---|---|---|---|---|
| TMA | PASS | PASS | PASS | 3 个 fresh-process replay；Triton tensor descriptor 实际执行 |
| multimem | topology_unavailable | PASS | N/A | PTX ISA 可编译；运行受 CUDA multicast/topology 门禁约束 |

## TMA 实测

- Driver `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED`：`1`。
- 4096×4096 BF16 copy+add：TMA candidate 中位数 `27.851 µs`；Torch add-out reference `19.501 µs`；比值 `1.428×`。
- 3 replay 均通过逐元素正确性、动态输入变化、CUDA Profiler 和编译产物指令门禁。
- SASS 明确包含 `UTMALDG.2D` 与 `UTMASTG.2D`。

## multimem 门禁

- Driver `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`：`0`；fabric handle：`1`；可见 GPU：`1`。
- 状态：`topology_unavailable`。`multimem.*` PTX 已由本机 ptxas 编译，但没有合法 CUDA multicast 映射，因此未运行。
- 编译后的 SM90 SASS 明确包含 `LDGMC.E.ADD.32.STRONG.SYS`。
- 普通指针不是 multimem address；对它发射 `multimem.*` 属于未定义行为，本测试严格禁止用这种方式伪造运行成功。

## 证据边界

- TMA 是本机真实执行与指令级证据，可作为 H200 SM90 feature PASS。
- multimem 是 ISA/toolchain PASS、当前单卡拓扑 runtime N/A；需要支持 switch multicast 的多 GPU/NVSwitch cell 才能形成运行评分。
- 原始 replay、Triton PTX/cubin/SASS、multimem PTX/cubin/SASS、Driver 属性与拓扑快照全部随 ZIP 提供。

参考：NVIDIA PTX ISA 的 `multimem.*` 定义与 CUDA Driver API multicast 管理。

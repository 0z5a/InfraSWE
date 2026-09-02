# 下一轮架构适配迭代建议

## P0：补齐硬件闭环

1. 在真实 MI300X 上执行 PyTorch 2.4.0 / ROCm 6.1 setup 与 formal runner，补齐
   AOTriton trace、七个经典 kernel、三轮 replay、评分和 provenance。
2. 在具备 CUDA multicast/NVSwitch 的多 GPU cell 上补 H200 multimem runtime；
   保持普通指针禁止执行 `multimem.*` 的安全门禁。
3. 为 B200 增加合法多 GPU Fabric cell，完成 SM100-Fabric，而不是把当前单卡 N/A
   改写为通过或 0 分。

## P1：统一硬件 profile

1. 新增实际使用的 `gpu-1x-sm90-h200-nvl-cuda128` profile。
2. 新增实际使用的 `gpu-1x-sm120-rtx-pro-5000-cuda128` profile；保留现有 2x
   SM120 PCIe profile 给未来多卡实验。
3. 将 driver、toolkit compiler、framework CUDA/ROCm runtime 和 topology 分开
   冻结，避免用单一 `runtime_version` 混淆编译器与框架运行时。

## P1：扩展同架构任务面

1. 在 B200/SM100 cell 补 FA/经典 kernel 评分，与 compiler-feature namespace
   分开发布。
2. 为 SM90/SM120 增加与 SM100 类似的 feature namespace 时，保持 task contract
   与硬件特性对应，不直接复用不适用的 opcode 门禁。
3. 多卡 SM80 profile 需要独立 collective/overlap 证据，不能继承单卡 kernel 分数。

## P2：结果与 schema 收敛

1. 把通用 kernel v0.3 和 SM100 feature v0.2 的公共字段统一到同一 envelope，
   保留各自 namespace 和公式版本。
2. 所有 runner 统一输出：capability fingerprint、artifact-set hash、三轮 replay、
   profiler sidecar、环境 freeze、逐文件 manifest 和 ZIP hash。
3. 在 CI 中增加 architecture matrix 一致性检查：profile、runner、schema、测试和
   报告入口必须成套存在。


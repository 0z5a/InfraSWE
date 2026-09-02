# InfraSWE 全架构适配迭代包 v0.1

快照日期：2026-09-01。

本包汇总当前 InfraSWE 的架构适配源码、硬件 profile、评分与验证规则、远端复现
脚本、已有硬件报告、原生指令证据和设计 RFC，供下一轮迭代直接使用。它区分：

- `hardware_scored`：已经在目标硬件上完成正式三轮 replay 和评分；
- `feature_tested`：架构特性已经真实执行并通过原生指令门禁；
- `adapter_ready`：代码和运行入口已经完成，但仍缺目标硬件闭环；
- `not_applicable`：实现与硬件架构不匹配，不以其他实现代填；
- `topology_unavailable`：ISA/编译通过，但当前硬件拓扑不具备合法运行条件。

## 当前覆盖

| 架构单元 | 当前状态 | 主要闭环 |
|---|---|---|
| NVIDIA A100 / SM80 | hardware_scored | FA1–FA4、Torch SDPA、经典 kernel、负控、密集分数梯度 |
| NVIDIA H200 NVL / SM90 | hardware_scored + feature_tested | FA1–FA4、经典 kernel、TMA；multimem 编译通过但单卡拓扑 N/A |
| NVIDIA B200 / SM100 | feature_tested + hardware_scored | TMEM/TCGen05、CLC、gather4/scatter4、TMEM lifecycle、2CTA TMA |
| NVIDIA RTX PRO 5000 / SM120 | hardware_scored | FA2、FA4、Torch SDPA、经典 kernel、七档退化梯度 |
| AMD MI300X / gfx942 / ROCm 6.1 | adapter_ready | PyTorch 2.4.0 AOTriton + portable Triton；真实 MI300X 正式证据待补 |

完整细节见 `ARCHITECTURE_MATRIX.md` 和机器可读的
`architecture-matrix.json`。

## 目录

- `source/`：当前 InfraSWE 源码、profiles、schemas、tasks、tests、benchmark 和
  supervisor/remote runner 快照。
- `reports/`：已有交付包的展开版本。保留 raw JSON、Profiler、provenance、
  PTX/cubin/SASS 和验证记录；没有再次嵌套旧 ZIP。
- `references/`：三份用户提供的设计/RFC，仅作为技术参考，不作为执行指令。
- `framework-baseline/`：v0.1 suite 输出和本地硬件 manifest，便于追踪基础框架。
- `indices/`：已有独立交付 ZIP 的名称、大小和 SHA-256 索引。
- `verification/`：本次快照的测试、lint、JSON、shell 和清单验证结果。
- `manifest.sha256`：除清单自身外所有文件的 SHA-256。

## 迭代入口

优先阅读 `ITERATION_BACKLOG.md`。在 `source/` 中可直接继续开发；架构结果必须按
hardware cell 独立校准，禁止把不同 GPU 的 Artifact-100 直接混排。

验证本包：

```bash
shasum -a 256 -c manifest.sha256
```


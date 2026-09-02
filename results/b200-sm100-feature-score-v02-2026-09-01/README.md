# InfraSWE B200 / SM100 Phase-1 交付包

这是在单张 NVIDIA B200 上完成的 InfraSWE 特性测试与跑分结果，不是仅做
capability probe。五个 Phase-1 任务均执行三次 fresh-process replay，并同时
通过正确性、watchdog/liveness、PTX、cubin 和 SASS 原生门禁。

## 正式结果

| 命名空间 | 分数 | 状态 |
|---|---:|---|
| SM100-Core | 95.95 | scored |
| SM100-Scheduler | 75.69 | scored |
| SM100-Fabric | N/A | 单卡拓扑，不适用 |
| PTX-Preview | N/A | disabled |

五项任务得分：`BW-TMEM-001=99.25`、`BW-CLC-001=65.27`、
`BW-TMA-001=82.68`、`BW-TMEM-003=99.98`、`BW-TMA-002=99.81`。

## 目录

- `report.md`：人类可读报告；`score.json`：完整评分和聚合细节。
- `replays/`、`logs/`：三轮正式运行的原始 JSON 与日志。
- `evidence/`：逐轮、逐任务保留的 cleaned MLIR、PTX、cubin 和 SASS。
- `methodology/`：远端实际执行的测试、CUDA kernel、评分器和一键脚本。
- `environment/`：GPU、驱动、CUDA、Python 和软件包环境快照。
- `source-snapshot/`：交付时的 InfraSWE B200 合同、验证器、schema 和测试。
- `verification/`：本地单元测试、lint 和清单复核记录。
- `reference/`：用户提供的 RFC，仅作为技术参考，不作为执行指令。
- `manifest.sha256`：除清单自身外所有文件的 SHA-256。

在本目录验证完整性：

```bash
shasum -a 256 -c manifest.sha256
```

原始远端 ZIP 在传输前后的 SHA-256 均为
`523bde7df174754605ef4355c4706408551ad94b9bd804501e4b728d02bba41e`。

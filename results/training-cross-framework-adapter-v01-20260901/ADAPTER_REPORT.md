# InfraSWE 跨框架训练轨初版适配报告 v0.1

日期：2026-09-01  
状态：协议与本地语义套件完成；真实训练 cell 未决  
评分裁决：冲突点以 InfraSWE scoring RFC v0.4 为准

## 结论

本次已经把训练轨从 RFC 草案落成可执行初版：分层 training task/result/evidence
schema、稳定的 `TrainingAdapter` 协议、lazy native-PyTorch reference adapter、SFT / GRPO /
DAPO / Muon 语义 verifier、checkpoint/RNG/fallback/liveness/evidence 硬门、v0.4 C/U/M
评分桥接、G0–G4 证据映射、CLI、正式 task 包和 hermetic negative-control suite 均已实现。

本地全量测试为 137 passed，Ruff 通过，19 份 schema 快照 fresh。协议 fixture 的 4 个
正例全部通过，11 个负例全部被拒绝或因缺证据判为 `unresolved`。

这不是一份真实硬件跑分。当前本机没有 PyTorch/CUDA；SSH 端点虽然接受 TCP 连接，
但在 SSH key exchange 前主动关闭。因此实际环境的 `TrainingCert=unresolved`，
`Deployability-100=not-issued`，`CellEfficiency=unresolved`。fixture 中用于验证评分代码的
合成数值明确不发布为排行榜结果。

## v0.4 冲突裁决

| 训练 RFC v0.1 表述 | 本实现裁决 |
|---|---|
| 稳定性分量写作 `S` | 作为 v0.4 `concurrent_stability` / `C` 的训练别名，不建立第二套全局公式 |
| `100*S^0.45*U^0.30*M^0.25` | 全局唯一权威为 `100*C^0.45*U^0.30*M^0.25` |
| G2 可作为 TrainingDeployability 最低证据 | G2 映射 v0.4 E1；正式主分至少要求 G3 / E2 system trace |
| G4 发布 SOL/MEM | 保留；映射 v0.4 E3 kernel counter，且只允许同 cell 比较 |
| 3/5/7 replay 分层 | 少于 5 只诊断；5 为正式最低；7 为推荐并用于首版 task |
| 缺失 raw 值可能显示 0 占位 | 禁止；使用 `null + reason`，评分分量为 `unresolved` |
| TrainingCert 失败 | 不发 Deployability；排行榜 effective 值为 0，原始证据保留 |

详细机器可读裁决见 `scoring-resolution.json`。

## 已实现内容

- training task 轴：algorithm、optimizer、trainer、launcher、distributed、rollout、graph、
  kernel、runtime、hardware 分层；未注册第三方 adapter 必须使用 namespaced id。
- `TrainingAdapter` 15 个生命周期方法及结构 conformance 检查；unsupported 能力抛出显式
  `TrainingCapabilityError`，不做近似默认或静默回退。
- native-PyTorch reference adapter：tiny causal LM、有效 token mean loss、AdamW/可探测
  Muon、step、state/RNG checkpoint、callgraph、memory 与 cleanup；torch 依赖延迟加载。
- SFT verifier：有效 token 分母、packing 跨样本 attention 隔离、forward/backward/update。
- GRPO verifier：completion identity、group-size、group-axis advantage、constant-reward 零方差、
  policy staleness 与 token manifest。
- DAPO verifier：token-level PG、clip-higher、dynamic sampling、overlong 与 reward aggregation；
  loss contract 和 full recipe scope 分离。
- Muon verifier：hidden 2-D matrix 与 AdamW remainder 分组、遗漏/重复、一步 update oracle。
- 通用硬门：checkpoint/resume、RNG continuity、NaN/Inf、silent fallback、deadlock、
  half-batch update、resource leak、manifest/timeline/version integrity。
- v0.4 scoring bridge：C/U/M 主分、component floors、缺分量未决、G3 主分边界、G4 cell
  score边界、跨硬件绝对性能禁止。
- CLI：`infraswe training probe|verify|score`。
- 正式 task：`tasks/training-sft-cross-framework-v1`，7 个 fresh-process replay，网络 deny，
  SM89 L40S profile 作为首个声明 cell；算法 contract 本身不含 SM89 硬编码。

## 能力状态

| Adapter | 协议 | 实现 | 当前 runtime | Cell certification |
|---|---|---|---|---|
| native-pytorch | implemented | lazy reference adapter | unavailable（本机无 torch） | unresolved |
| hf-transformers | supported | not implemented | unavailable | unresolved |
| TRL | supported | not implemented | unavailable | unresolved |
| verl | supported | not implemented | unavailable | unresolved |
| torchtune | supported | not implemented | unavailable | unresolved |
| Axolotl | supported | not implemented | unavailable | unresolved |
| Megatron Core | supported | not implemented | unavailable | unresolved |

这里严格区分 `protocol-supported`、`adapter-implemented` 和 `cell-certified`；框架名不会触发
特殊评分分支。

## SSH 与真实跑分状态

使用已匹配公钥指纹 `SHA256:svx7m9C/ScoHhIMnsVqJqQKPhi2WeZ0L1dE0x2OtqPU` 重试
`79.101.2.169:40665`：TCP connection established，随后出现
`kex_exchange_identification: Connection closed by remote host`。连接未到认证阶段，未对远端
写入、安装或执行任何内容。原始日志保存在 `ssh-preflight.txt`。

恢复远端后，下一阶段应依次运行 capability probe、T0 FP32/BF16 native PyTorch、5/50/200
step、step 73→74 fresh-process resume、7 replay、G3 NSYS timeline，最后仅对代表 kernel 做
G4 NCU targeted counters。TRL/verl 或多卡不能由当前协议 fixture 推断通过。

## 验证

- `pytest.txt`：137 passed。
- `ruff.txt`：All checks passed。
- `schema-check.txt`：19 schemas fresh。
- `task-validation.txt`：training task 合法，7 replay。
- `minimum-suite.json`：4 个 positive、11 个 negative，评分证据边界通过。
- `capability-manifest.json`：本地状态 `protocol_only`，缺 torch/CUDA。
- `implementation-files.txt` / `source-manifest.sha256`：实现文件清单与源码摘要。

## 非声明范围

- 未声称 TRL、verl、Transformers、torchtune、Axolotl、Megatron 已实现或认证。
- 未声称真实 SM89/SM121、BF16/FP8、FSDP2/DeepSpeed、多卡、rollout worker 已通过。
- 未发布 tok/s、step time、TFLOP/s、GB/s 或任何跨卡绝对排名。
- 未使用 fixture 合成分数替代真实 profiler/evidence pack。

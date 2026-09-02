# 历史 PR 盲测中间状态（2026-09-01）

## 盲态完整性

- 预测锁尚未冻结。
- 尚未查询任何候选的 `merged`、review、review comments 或 CI 结论。
- 合并真值与 reviewer 反馈仍未揭晓。
- 原计划为 L40S/SM89；用户提供的新实例实测为 2×A100-PCIE-40GB/SM80，已生成显式 test-plan revision，未把 A100 结果冒充 L40S 结果。

## 已形成的候选证据

| Case | Head | Base 对照 | 当前解释 |
|---|---|---|---|
| `sglang-pr-3121` | `sgl-kernel` 构建成功；变更测试 144/144 通过 | head 已通过，按预注册策略不强制跑 base | A100/SM80 machine-check pass |
| `flashinfer-pr-624` | AOT URI probe 通过：336 URI、0 个 mask-encoded prefill URI | probe 失败：624 URI、432 个 mask-encoded prefill URI | head 修复被探针区分 |
| `flashinfer-pr-632` | 20/20 JIT 模板严格渲染；suffix/template 一致 | 缺 9 个 mask split suffix，probe 失败 | head 模板拆分契约通过 |
| `flashattention-pr-1326` | CUDA 12.4/12.5 均映射 `cu124`，CI 矩阵为 12.4.1 | CUDA 12 映射 `cu123`，CI 矩阵为 12.3.2 | portable exact-SHA probe 已区分；待远端复核落盘 |

12 个候选的 checkout、GraphQL-vs-git path parity、diff、Python AST/YAML/JSON 静态检查均已通过。

## 新增失败 PR 复核约束

失败反馈样本必须满足：

1. PR 为 `closed` 且未合并；
2. 存在人类 reviewer/maintainer 反馈；
3. 至少有 inline 技术评论、带正文的 `CHANGES_REQUESTED`，或 maintainer 明确 PR 反馈；
4. review 内容只能在机器预测与分数冻结后读取；
5. reviewer 反馈与冻结失败码按“完全对应 / 部分对应 / 未覆盖 / 机器额外发现”报告，不事后改分。

对应的 post-lock review evidence 模型、拉取工具与 JSON Schema 已实现。

## 当前阻塞

`47.160.150.235:41106` 仍可建立 TCP，但服务端在发送 SSH banner 前关闭连接；`ssh` 与 `ssh-keyscan` 均复现。该故障发生在认证之前，不是候选失败。

远端 `/workspace` 非持久，因此 SGLang #3121 的原始 build/test log 仍需在入口恢复后第一时间回收。剩余 GPU/build 项不得在无 runner 时伪造为通过或失败。

## 本地回归

- Ruff：通过。
- Pytest：162 passed。
- JSON Schema：34/34 fresh。

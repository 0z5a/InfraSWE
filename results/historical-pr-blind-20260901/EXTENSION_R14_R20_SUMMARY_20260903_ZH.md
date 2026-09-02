# R14–R20：200 个历史 PR 迭代增测总结

## 完成情况

本轮按用户要求完成 `50 communication + 50 training + 100 inference`，共 200 个历史 PR。
R14–R19 每组 30 个，R20 为对齐总量的终局 20 个；每组都严格执行“metadata-only 选样与
test-plan 冻结 → 获取清洗正文和源码 → 静态/远端测试 → pre-reveal 判断锁 → 结果揭晓 → 审计 →
下一组前瞻迭代”。R13 保持 29 例的既有边界不变，新增迭代从 R14 的 30 例组开始。

推理 100 例按 `vLLM / SGLang / TensorRT-LLM / FlashInfer` 各 25 例收口。所有判断使用
`accept/check/reject`，不再使用 `revise`。

## 总体结果

| 指标 | 同队列旧路数 | R14–R20 | 变化 |
|---|---:|---:|---:|
| Exact label | 68/200（34.0%） | 98/200（49.0%） | +15.0 pp |
| Binary accept/non-accept | 112/200（56.0%） | 112/200（56.0%） | 0 pp |
| 81 个冻结非 accept 的 exact | 9/81（11.1%） | 39/81（48.1%） | +37.0 pp |
| Reject precision | N/A | 36/69（52.2%） | 可测 |
| Check precision | 9/81（11.1%） | 3/12（25.0%） | +13.9 pp |
| Merged accept recall | — | 59/87（67.8%） | — |

整体 exact 提升 30 例，且二分类完全不变，说明收益确实集中在用户要求的 check/reject 区分，
并非通过改变 accept/non-accept 边界制造。终局 R20 达到 `14/20` exact、75% reject precision，
同时把 merged accept recall 恢复到 `11/11`。

## 分轮结果

| 轮次 | 类型 | 当前 exact | 旧路数 exact | 提升 |
|---|---|---:|---:|---:|
| R14 | 30 communication | 11/30（36.7%） | 6/30（20.0%） | +16.7 pp |
| R15 | 20 communication + 10 training | 19/30（63.3%） | 11/30（36.7%） | +26.7 pp |
| R16 | 30 training | 12/30（40.0%） | 9/30（30.0%） | +10.0 pp |
| R17 | 10 training + 20 inference | 13/30（43.3%） | 9/30（30.0%） | +13.3 pp |
| R18 | 30 inference | 14/30（46.7%） | 12/30（40.0%） | +6.7 pp |
| R19 | 30 inference | 15/30（50.0%） | 9/30（30.0%） | +20.0 pp |
| R20 | 20 inference | 14/20（70.0%） | 12/20（60.0%） | +10.0 pp |

七组都达到“有提升再提交”的门槛。R18–R20 的详细证据、逐例结果和哈希绑定分别见各自报告。

## 可复用分类经验

1. 技术正确性和历史 disposition 必须双输出。源码与测试能较好回答 contract 是否成立，却不能
   稳定观察 ownership、优先级、重复实现和真实 review 活跃度。
2. `check` 只适用于冻结前允许证据中存在独立、非作者人类 review/handoff 的情况。若协议禁止
   该通道，应报告不可观察；机器人 QA 或作者自述不能替代 oracle。
3. 候选自带的精确边界断言失败是强 reject 信号；无关导入、旧 fixture、目标硬件或生成 bindings
   缺口则必须与候选失败分开。
4. 成熟 final-head 若有正负控制及可达集成路径或独立窄不变量，应优先 technical accept；测试语言、
   C++ 路径或本地 capability gap 不能机械转成 reject。
5. 每轮单独审计 merged accept recall，防止扩大 reject 数量带来虚假的 exact 改善。

## 限制

本实验是项目分层的历史样本，不是所有基础设施仓库的随机样本。盲测策略刻意屏蔽了 oracle 所需的
部分 review 活动，因此 check 的完美召回在当前证据边界下结构上不可达。实验也没有使用加权分数；
标签不能解释为代码质量百分制，旧的“merged 必须 85 分以上”门槛在这里不可审计。

# 快照验证说明

正式验收使用当前项目 venv 显式加入 PATH 后执行：

```bash
PATH=/Users/0z5a/Documents/infra/infraswe/.venv/bin:$PATH \
  .venv/bin/pytest -q
```

结果为 91 passed，见 `pytest-full.txt`。

第一次直接调用 `.venv/bin/pytest` 时，4 个 runner 测试的子进程找不到命令名
`python`，其余 87 项通过。该诊断完整保存在
`pytest-full-initial-path-failure.txt`；加入 venv PATH 后同一测试集全部通过，说明是
调用环境问题而不是代码/架构适配失败。

其余记录：

- `ruff.txt`：源码、测试和 kernel frontier runner 的静态检查；
- `bash-syntax.txt` / `bash-files.txt`：全部 shell 入口的语法检查；
- `json-validation.txt` / `json-files.txt`：包内 JSON 解析检查；
- `toml-validation.txt`：profile 与项目 TOML 解析检查。
- `matrix-routing.txt`：架构矩阵引用的 report/profile 路径检查。
- `source-snapshot-diff.txt`：包内源码与当前工作目录的内容级比对。
- `git-diff-check.txt`：工作区文本补丁空白错误检查。

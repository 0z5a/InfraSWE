# Default Draft catalog v0.5

This catalog contains ten independent, project-conditioned Draft profiles. The initial four
inference/attention profiles are joined by six pinned GEMM and training profiles. The profiles are
not merged into one cross-project score.

Resolution order is frozen as follows:

1. an explicitly supplied local Draft;
2. an explicitly supplied remote Git Draft (`repository + revision + path`);
3. this built-in catalog when neither explicit source is set.

Within the default catalog, exact repository or entrypoint aliases select a profile. If no alias
matches, the frozen order starts with `vllm`, `sglang`, `flash-attention`, `flashinfer`, followed by
`cutlass-cute`, `liger-kernel`, `deepgemm`, `megatron-core`, `torchtitan`, and `verl`; the resolution
emits `DEFAULT_TARGET_SELECTED_BY_PRIORITY` and must report vLLM as the default target. It must not
call the result universally best or compare its ProjectFit against another project cell.

| Project | Catalog profile | Pinned upstream revision |
|---|---|---|
| vLLM | `vllm-kernel-integration-v1` | `40824284bcb2f50047a48307ed39ce441bb15b0b` |
| SGLang | `sglang-runtime-kernel-v1` | `4c2c169e6ba15aee5408b250ce25ff7e73388d9b` |
| FlashAttention | `flash-attention-kernel-v1` | `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` |
| FlashInfer | `flashinfer-kernel-library-v1` | `9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea` |
| CUTLASS / CuTe | `cutlass-cute-kernel-library-v1` | `dc45f979ae336a235da1676b311f35efeb30149a` |
| Liger-Kernel | `liger-training-fused-kernel-v1` | `e6a81bb0c34f31ca7806d0c2b72f6d66b0542694` |
| DeepGEMM | `deepgemm-moe-gemm-kernel-v1` | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| Megatron-Core | `megatron-core-training-kernel-host-v1` | `3c04d2bd2255c9652a687c3d5a5b9636467696db` |
| TorchTitan | `torchtitan-pytorch-native-training-host-v1` | `496b11d43860bb8d27b54568c76db6310ae7f55e` |
| verl | `verl-posttraining-rollout-host-v1` | `c2429f29a25d573f63d9bcc29e7ceb690817dce9` |

Each profile binds eight inspectable contract artifacts: API/ABI, lifecycle, build/test matrix,
dependency policy, fallback policy, deployment workload portfolio, performance targets, and
maintainability probes. Source URLs inside every artifact are revision-pinned. Runtime resolution
never fetches the latest upstream `main` to mutate the acceptance contract.

Status is deliberately `proposed`. These profiles were machine-extracted from upstream project
materials and have not been signed by an authorized project-profile maintainer. They can seed a
D3 contract proposal, but cannot enter D4, Seal, official scoring, or a leaderboard until a human
review record is bound and the normal v0.5 lifecycle is completed.

Regenerate and verify the materialized catalog with:

```bash
infraswe draft defaults --output catalog/default-drafts-v0.5
pytest -q tests/test_draft_v05.py
```

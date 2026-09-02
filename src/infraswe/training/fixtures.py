from __future__ import annotations

from infraswe.models.training import TrainingEvidenceBundle

MANIFEST_DIGEST = "sha256:" + "a" * 64
RAW_DIGEST = "sha256:" + "b" * 64


def _comparison(quantity: str) -> dict:
    return {
        "reference": [0.125, -0.25, 0.5],
        "candidate": [0.125, -0.25, 0.5],
        "atol": 1e-8,
        "rtol": 1e-7,
        "quantity": quantity,
        "dtype": "fp32",
        "sequence_bucket": 128,
        "distributed_mode": "single",
    }


def _base(*, algorithm: str, optimizer: str, scope: str) -> dict:
    return {
        "schema_version": "0.1",
        "task_id": f"training-{algorithm}-contract-v1",
        "algorithm": algorithm,
        "optimizer": optimizer,
        "certification_scope": scope,
        "adapter_id": "native-pytorch",
        "framework_version": "fixture",
        "framework_stack_id": "native-pytorch@fixture",
        "hardware_cell_id": "sha256:" + "c" * 64,
        "implementation_bundle_id": "sha256:" + "d" * 64,
        "evidence_manifest_sha256": MANIFEST_DIGEST,
        "normalized_config": {
            "global_batch_tokens": 256,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "sequence_length_policy": "packed-variable",
            "precision": "fp32",
            "loss_reduction": "valid-target-token-mean",
            "gradient_clipping": "global-l2",
            "optimizer": optimizer,
            "learning_rate_schedule": "cosine",
            "activation_checkpointing": False,
            "seed_bundle": {"model": 101, "data": 102, "sampling": 103, "dropout": 104},
        },
        "forward": _comparison("logits"),
        "backward": _comparison("gradients"),
        "optimizer_update": _comparison("parameters-after-update"),
        "checkpoint": {
            "saved_components": ["weights", "optimizer", "scheduler", "data_cursor"],
            "restored_components": ["weights", "optimizer", "scheduler", "data_cursor"],
            "next_step_comparison": _comparison("resume-next-step"),
            "rng_streams_restored": ["data_rng", "dropout_rng", "sampling_rng"],
            "fresh_process": True,
        },
        "runtime": {
            "loss_values": [1.2, 1.1, 1.0],
            "silent_fallback_count": 0,
            "declared_fallbacks": [],
            "deadlock": False,
            "watchdog_passed": True,
            "resource_leaks": [],
            "half_batch_updates": 0,
        },
        "integrity": {
            "manifest_sha256": MANIFEST_DIGEST,
            "raw_evidence_digests": [RAW_DIGEST],
            "timeline_consistent": True,
            "versions_exact": True,
        },
    }


def sft_reference_bundle(*, muon: bool = False) -> TrainingEvidenceBundle:
    optimizer = "muon-plus-adamw" if muon else "adamw"
    payload = _base(algorithm="sft", optimizer=optimizer, scope="sft-contract")
    payload["sft"] = {
        "token_losses": [0.2, 0.4, 0.8, 0.6],
        "target_mask": [False, True, False, True],
        "observed_loss": 0.5,
        "observed_denominator": 2,
        "packed_sample_ids": ["sample-a", "sample-a", "sample-b", "sample-b"],
        "observed_attention_edges": [[0, 0], [1, 0], [1, 1], [2, 2], [3, 2], [3, 3]],
    }
    if muon:
        payload["muon"] = {
            "trainable_parameters": [
                "blocks.0.proj.weight",
                "token_embedding.weight",
                "lm_head.weight",
                "blocks.0.norm.weight",
            ],
            "parameter_groups": [
                {
                    "name": "blocks.0.proj.weight",
                    "shape": [4, 4],
                    "semantic_role": "hidden-matrix",
                    "group_id": "muon-hidden",
                    "optimizer": "muon",
                    "state_shape": [4, 4],
                    "update_count": 1,
                },
                {
                    "name": "token_embedding.weight",
                    "shape": [16, 4],
                    "semantic_role": "embedding",
                    "group_id": "adamw-remaining",
                    "optimizer": "adamw",
                    "state_shape": [16, 4],
                    "update_count": 1,
                },
                {
                    "name": "lm_head.weight",
                    "shape": [16, 4],
                    "semantic_role": "output-head",
                    "group_id": "adamw-remaining",
                    "optimizer": "adamw",
                    "state_shape": [16, 4],
                    "update_count": 1,
                },
                {
                    "name": "blocks.0.norm.weight",
                    "shape": [4],
                    "semantic_role": "norm",
                    "group_id": "adamw-remaining",
                    "optimizer": "adamw",
                    "state_shape": [4],
                    "update_count": 1,
                },
            ],
            "newton_schulz_iterations": 5,
            "coefficients_id": "quintic-v1",
            "normalization_id": "spectral-upper-bound-v1",
            "epsilon": 1e-7,
            "update_comparison": _comparison("muon-one-step-update"),
        }
    return TrainingEvidenceBundle.model_validate(payload)


def _rollout_samples() -> list[dict]:
    common = {
        "policy_version": 5,
        "train_policy_version": 5,
        "token_ids": [10, 11],
        "old_log_probs": [-0.2, -0.3],
        "valid_token_mask": [True, True],
    }
    return [
        {
            **common,
            "prompt_id": "prompt-a",
            "group_id": "group-a",
            "sample_id": "a-0",
            "sampling_seed": 100,
            "reward": 1.0,
            "observed_advantage": -1.0,
        },
        {
            **common,
            "prompt_id": "prompt-a",
            "group_id": "group-a",
            "sample_id": "a-1",
            "sampling_seed": 101,
            "reward": 3.0,
            "observed_advantage": 1.0,
        },
        {
            **common,
            "prompt_id": "prompt-b",
            "group_id": "group-b",
            "sample_id": "b-0",
            "sampling_seed": 102,
            "reward": 2.0,
            "observed_advantage": 0.0,
        },
        {
            **common,
            "prompt_id": "prompt-b",
            "group_id": "group-b",
            "sample_id": "b-1",
            "sampling_seed": 103,
            "reward": 2.0,
            "observed_advantage": 0.0,
        },
    ]


def grpo_reference_bundle() -> TrainingEvidenceBundle:
    payload = _base(algorithm="grpo", optimizer="adamw", scope="grpo-contract")
    payload["grpo"] = {
        "samples": _rollout_samples(),
        "expected_group_size": 2,
        "advantage_epsilon": 1e-8,
        "advantage_tolerance": 1e-6,
        "max_policy_staleness": 1,
        "kl_definition": "reverse-kl-token-masked-v1",
        "kl_sign": "penalty-positive",
    }
    return TrainingEvidenceBundle.model_validate(payload)


def dapo_reference_bundle() -> TrainingEvidenceBundle:
    payload = _base(algorithm="dapo", optimizer="adamw", scope="dapo-recipe-contract")
    payload["grpo"] = {
        "samples": _rollout_samples(),
        "expected_group_size": 2,
        "advantage_epsilon": 1e-8,
        "advantage_tolerance": 1e-6,
        "max_policy_staleness": 1,
        "kl_definition": "reverse-kl-token-masked-v1",
        "kl_sign": "penalty-positive",
    }
    payload["dapo"] = {
        "token_level_policy_gradient": True,
        "asymmetric_clip_higher": True,
        "dynamic_sampling": True,
        "overlong_policy_exact": True,
        "soft_overlong_punishment_exact": True,
        "reward_aggregation_exact": True,
    }
    return TrainingEvidenceBundle.model_validate(payload)

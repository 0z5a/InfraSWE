from __future__ import annotations

import hashlib
import hmac
import json

from infraswe.kernel.models import RoleResult


def canonical_role_payload(result: RoleResult) -> bytes:
    payload = result.model_dump(
        mode="json",
        exclude={"result_sha256", "signature"},
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_role_result(result: RoleResult, signing_key: bytes) -> RoleResult:
    digest = "sha256:" + hashlib.sha256(canonical_role_payload(result)).hexdigest()
    message = f"{digest}\n{result.role_instance_id}\n{result.identity.task_package_sha256}".encode()
    signature = "hmac-sha256:" + hmac.new(signing_key, message, hashlib.sha256).hexdigest()
    return result.model_copy(update={"result_sha256": digest, "signature": signature})


def verify_role_result(result: RoleResult, signing_key: bytes) -> bool:
    if not result.result_sha256.startswith("sha256:"):
        return False
    if not result.signature.startswith("hmac-sha256:"):
        return False
    unsigned = result.model_copy(update={"result_sha256": "", "signature": ""})
    expected = seal_role_result(unsigned, signing_key)
    digest_matches = hmac.compare_digest(result.result_sha256, expected.result_sha256)
    signature_matches = hmac.compare_digest(result.signature, expected.signature)
    return digest_matches and signature_matches

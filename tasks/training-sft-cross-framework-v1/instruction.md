# Cross-framework SFT adapter task

Implement the adapter manifest in `fixture/repo/adapter_manifest.json` without changing the
semantic contract. The manifest must retain assistant-only masking, packed-sample isolation,
valid-target-token mean reduction, exact optimizer-step boundaries, explicit fallback reporting,
and fresh-process checkpoint/RNG restoration.

Framework defaults that cannot express these semantics must be reported as `unsupported`; they
must not be approximated silently.

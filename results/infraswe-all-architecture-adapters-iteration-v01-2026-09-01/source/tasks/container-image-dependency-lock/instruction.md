# Make image resolution immutable and architecture-safe

The image resolver currently accepts the first registry candidate, including mutable tags,
wrong-architecture layers, and conflicting transitive packages. Repair `resolver.py` and
`lock_policy.json` while preserving `resolve_image(request, candidates, policy) -> dict`.

The fixed resolver must:

- require `sha256:<64 lowercase hex>` digests for the root image and every dependency;
- match image name, version, and requested architecture exactly;
- reject duplicate dependency names with conflicting versions or digests;
- choose deterministically independent of registry response order;
- return an explicit blocked plan when no immutable candidate is safe;
- never use `latest`, an unpinned layer, or a cross-architecture fallback;
- reject malformed requests and avoid mutating inputs.

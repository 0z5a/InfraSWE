# Select native CUDA extension architecture targets

The build policy hard-codes an SM target from a different GPU generation and silently relies on
unsupported PTX fallback. Repair `arch_policy.py` and `build_config.json`, preserving
`select_targets(device_sms, toolkit, config) -> dict`.

The fixed policy must emit one deterministic native SASS target for every distinct visible device
SM, but only when the toolkit reports support for every target. Unsupported or malformed fleets
must be blocked explicitly; cross-generation and PTX fallback are forbidden for this native
extension. Target order must not depend on device enumeration, configuration must use visible-device
discovery, and inputs must not be mutated.

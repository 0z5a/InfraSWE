from __future__ import annotations

import json
from pathlib import Path

path = Path("deployment.json")
deployment = json.loads(path.read_text(encoding="utf-8"))
spec = deployment["spec"]
spec["readinessProbe"] = {
    "path": "/readyz",
    "initialDelaySeconds": 2,
    "periodSeconds": 1,
}
spec["strategy"] = {"maxUnavailable": 0, "maxSurge": 1}
spec["terminationGracePeriodSeconds"] = 10
spec["rollbackEnabled"] = True
path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

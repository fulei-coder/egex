from __future__ import annotations

import sys
import types


def register_with_lerobot() -> None:
    """Expose the local EgExo policy under lerobot.policies.egexo_smolvla.

    This keeps the standard `python -m lerobot.scripts.lerobot_train` path
    working while the actual implementation lives under `realman_vla.*`.
    """

    try:
        import lerobot.policies  # noqa: F401
        from realman_vla.policies.egexo_smolvla import configuration_egexo_smolvla
        from realman_vla.policies.egexo_smolvla import modeling_egexo_smolvla
    except Exception:
        return

    package_name = "lerobot.policies.egexo_smolvla"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []  # mark as package-like for import machinery
        sys.modules[package_name] = package

    sys.modules[f"{package_name}.configuration_egexo_smolvla"] = configuration_egexo_smolvla
    sys.modules[f"{package_name}.modeling_egexo_smolvla"] = modeling_egexo_smolvla

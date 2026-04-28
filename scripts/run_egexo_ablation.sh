#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE_CONFIG="${1:-configs/egexo_smolvla_realman.yaml}"
MODE="${2:-print}"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "❌ Base config not found: ${BASE_CONFIG}"
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ABLT_DIR="$(mktemp -d /tmp/egexo_ablation.XXXXXX)"
export EGEXO_BASE_CONFIG="${BASE_CONFIG}"
export EGEXO_ABLATION_DIR="${ABLT_DIR}"

python3 - <<'PY'
import copy
import os
from pathlib import Path
import yaml

base_config = Path(os.environ["EGEXO_BASE_CONFIG"]).resolve()
out_dir = Path(os.environ["EGEXO_ABLATION_DIR"]).resolve()
with base_config.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

variants = {
    "cam_high_only": {
        "ablation": {"use_exo": True, "use_ego": False, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False},
        "policy": {"egexo": {"use_exo": True, "use_ego": False, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False}},
    },
    "cam_wrist_only": {
        "ablation": {"use_exo": False, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False},
        "policy": {"egexo": {"use_exo": False, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False}},
    },
    "naive_concat": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False, "use_view_embedding": False},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False, "use_view_embedding": False}},
    },
    "asymmetric": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False, "use_view_embedding": True},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": False, "use_grounding_loss": False, "use_phase_head": False, "use_dual_action_head": False, "use_view_embedding": True}},
    },
    "grounding": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": False, "use_dual_action_head": False},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": False, "use_dual_action_head": False}},
    },
    "phase_head": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": False},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": False}},
    },
    "full": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": True, "use_view_embedding": True},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": True, "use_view_embedding": True}},
    },
    "full_no_aug": {
        "ablation": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": True, "use_view_embedding": True},
        "policy": {"egexo": {"use_exo": True, "use_ego": True, "use_crossview_grounding": True, "use_grounding_loss": True, "use_phase_head": True, "use_dual_action_head": True, "use_view_embedding": True}},
    },
}

out_dir.mkdir(parents=True, exist_ok=True)
manifest = []
for name, update in variants.items():
    variant = copy.deepcopy(cfg)
    variant.setdefault("policy", {})
    variant.setdefault("training", {})
    variant.setdefault("ablation", {})
    for key, value in update.get("ablation", {}).items():
        variant["ablation"][key] = value
    for key, value in update.get("policy", {}).items():
        if isinstance(value, dict):
            variant["policy"].setdefault(key, {})
            variant["policy"][key].update(value)
        else:
            variant["policy"][key] = value

    variant["policy"]["repo_id"] = f"local/egexo_smolvla_{name}"
    variant["training"]["output_dir"] = f"outputs/ablation/{name}"
    variant["training"]["run_name"] = name
    variant["training"]["resume"] = False
    if name == "full_no_aug":
        variant["training"]["disable_data_augmentation"] = True
    else:
        variant["training"].pop("disable_data_augmentation", None)

    out_path = out_dir / f"{name}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(variant, f, sort_keys=False, allow_unicode=True)
    manifest.append({"name": name, "config": str(out_path), "output_dir": variant["training"]["output_dir"]})

manifest_path = out_dir / "manifest.yaml"
with manifest_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump({"base_config": str(base_config), "variants": manifest}, f, sort_keys=False, allow_unicode=True)
print(manifest_path)
PY

MANIFEST_PATH="${ABLT_DIR}/manifest.yaml"
echo "Ablation manifest: ${MANIFEST_PATH}"

if [ "${MODE}" = "print" ]; then
    echo ""
    echo "Generated variants:"
    sed -n '1,240p' "${MANIFEST_PATH}"
    echo ""
    echo "Run a single variant:"
    echo "  python3 -m realman_vla.policies.egexo_smolvla.train --config ${ABLT_DIR}/full.yaml"
    echo ""
    echo "Run all variants:"
    echo "  bash scripts/run_egexo_ablation.sh ${BASE_CONFIG} run"
    exit 0
fi

if [ "${MODE}" = "run" ]; then
    while IFS= read -r cfg_path; do
        [ -z "${cfg_path}" ] && continue
        echo "============================================"
        echo "Running ablation config: ${cfg_path}"
        echo "============================================"
        python3 -m realman_vla.policies.egexo_smolvla.train --config "${cfg_path}"
    done < <(python3 - <<'PY'
import os
from pathlib import Path
import yaml
manifest = yaml.safe_load(Path(os.environ["EGEXO_ABLATION_DIR"]).joinpath("manifest.yaml").read_text(encoding="utf-8"))
for item in manifest["variants"]:
    print(item["config"])
PY
)
    exit 0
fi

echo "❌ Unknown mode: ${MODE}"
echo "Usage:"
echo "  bash scripts/run_egexo_ablation.sh [base_config] [print|run]"
exit 1

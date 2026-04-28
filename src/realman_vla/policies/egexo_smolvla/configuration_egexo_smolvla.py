from __future__ import annotations

from dataclasses import dataclass, field

try:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
except Exception:  # pragma: no cover - fallback for environments without lerobot
    @dataclass
    class SmolVLAConfig:  # type: ignore[override]
        type: str = "smolvla"
        repo_id: str | None = None


@dataclass
class EgExoSmolVLAConfig(SmolVLAConfig):
    type: str = "egexo_smolvla"
    base_policy_type: str = "smolvla"
    image_keys: dict = field(
        default_factory=lambda: {
            "exo": "observation.images.cam_high",
            "ego": "observation.images.cam_wrist",
        }
    )
    state_key: str = "observation.state"
    ee_pose_key: str = "observation.ee_pose"
    ego_roi_key: str = "observation.grounding.ego_roi"
    grounding_valid_key: str = "observation.grounding.valid"
    phase_key: str = "observation.phase"
    egexo: dict = field(
        default_factory=lambda: {
            "use_exo": True,
            "use_ego": True,
            "use_view_embedding": True,
            "use_crossview_grounding": True,
            "grounding_mode": "soft_mask",
            "use_grounding_loss": True,
            "use_phase_head": True,
            "phase_mode": "dual_head",
            "action_mixing": "soft",
            "use_phase_embedding": True,
            "phase_embedding_dim": 32,
            "phase_embedding_scale": 0.25,
            "phase_gate_hidden_dim": 64,
            "phase_gate_use_ee_pose": True,
            "use_dual_action_head": True,
            "dual_action_head_hidden_dim": 128,
            "dual_action_head_use_ee_pose": True,
        }
    )
    loss: dict = field(
        default_factory=lambda: {
            "action_weight": 1.0,
            "grounding_weight": 0.1,
            "phase_weight": 0.05,
            "cross_view_consistency_weight": 0.0,
            "dual_action_weight": 0.1,
        }
    )

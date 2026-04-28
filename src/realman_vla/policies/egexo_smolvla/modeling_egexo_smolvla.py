from __future__ import annotations

import copy

try:
    import torch
except Exception:  # pragma: no cover - local shell may not have torch
    torch = None
if torch is not None:  # pragma: no branch
    import torch.nn as nn
else:  # pragma: no cover - local shell may not have torch
    nn = None

from .configuration_egexo_smolvla import EgExoSmolVLAConfig

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except Exception:  # pragma: no cover - fallback for environments without lerobot
    class SmolVLAPolicy:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EgExoSmolVLAPolicy requires lerobot to be installed in the active environment."
            )


class EgExoSmolVLAPolicy(SmolVLAPolicy):
    """Minimal stage-A Ego-Exo wrapper on top of SmolVLA.

    This version keeps the original SmolVLA action expert intact and only
    injects a lightweight wrist-image soft mask driven by the projected ego ROI.
    The extra ego/exo metadata keys are tolerated but not required by the base
    policy, which keeps training and inference compatible with existing code.
    """

    config_class = EgExoSmolVLAConfig
    name = "egexo_smolvla"

    def __init__(self, *args, **kwargs):  # pragma: no cover - exercised in training env
        super().__init__(*args, **kwargs)
        self._init_phase_embedding_modules()
        self._init_phase_gate_modules()
        self._init_dual_action_head_modules()

    def _infer_state_dim(self) -> int:
        config = getattr(self, "config", None)
        if config is None:
            return 0
        state_key = getattr(config, "state_key", "observation.state")
        input_features = getattr(config, "input_features", {}) or {}
        feature = input_features.get(state_key)
        if feature is None:
            return 0
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if not shape:
            return 0
        try:
            return int(shape[-1])
        except Exception:
            return 0

    def _infer_action_dim(self) -> int:
        config = getattr(self, "config", None)
        if config is None:
            return 0
        output_features = getattr(config, "output_features", {}) or {}
        feature = output_features.get("action")
        if feature is None:
            return 0
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if not shape:
            return 0
        try:
            return int(shape[-1])
        except Exception:
            return 0

    def _init_phase_embedding_modules(self) -> None:
        self.phase_embedding = None
        self.phase_embedding_scale = 0.0
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if torch is None or nn is None or not egexo_cfg:
            return
        if not egexo_cfg.get("use_phase_embedding", True):
            return

        state_dim = self._infer_state_dim()
        if state_dim <= 0:
            return

        hidden_dim = int(egexo_cfg.get("phase_embedding_dim", min(32, state_dim)))
        hidden_dim = max(1, hidden_dim)
        self.phase_embedding_scale = float(egexo_cfg.get("phase_embedding_scale", 0.25))
        self.phase_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),
        )

    def _init_phase_gate_modules(self) -> None:
        self.phase_gate = None
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if torch is None or nn is None or not egexo_cfg:
            return
        if not egexo_cfg.get("use_phase_head", True):
            return

        state_dim = self._infer_state_dim()
        if state_dim <= 0:
            return

        input_dim = state_dim
        if egexo_cfg.get("phase_gate_use_ee_pose", True):
            input_dim += 6
        hidden_dim = int(egexo_cfg.get("phase_gate_hidden_dim", max(32, state_dim)))
        hidden_dim = max(1, hidden_dim)
        self.phase_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _init_dual_action_head_modules(self) -> None:
        self.transport_action_head = None
        self.contact_action_head = None
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if torch is None or nn is None or not egexo_cfg:
            return
        if not egexo_cfg.get("use_dual_action_head", True):
            return

        state_dim = self._infer_state_dim()
        action_dim = self._infer_action_dim()
        if state_dim <= 0 or action_dim <= 0:
            return

        input_dim = state_dim
        if egexo_cfg.get("dual_action_head_use_ee_pose", True):
            input_dim += 6
        hidden_dim = int(egexo_cfg.get("dual_action_head_hidden_dim", max(64, action_dim * 4)))
        hidden_dim = max(1, hidden_dim)

        def make_head():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        self.transport_action_head = make_head()
        self.contact_action_head = make_head()

    def _apply_ego_soft_mask(self, batch: dict) -> dict:
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if not egexo_cfg or not egexo_cfg.get("use_crossview_grounding", True):
            return batch

        image_key = getattr(config, "image_keys", {}).get("ego", "observation.images.cam_wrist")
        roi_key = getattr(config, "ego_roi_key", "observation.grounding.ego_roi")
        valid_key = getattr(config, "grounding_valid_key", "observation.grounding.valid")

        image = batch.get(image_key)
        roi = batch.get(roi_key)
        valid = batch.get(valid_key)
        if image is None or roi is None or valid is None:
            return batch
        if torch is None:
            return batch
        if not torch.is_tensor(image) or image.ndim != 4:
            return batch
        if not torch.is_tensor(roi) or roi.ndim != 2 or roi.shape[-1] != 4:
            return batch
        if not torch.is_tensor(valid):
            return batch

        mask = torch.full(
            (image.shape[0], 1, image.shape[-2], image.shape[-1]),
            0.35,
            device=image.device,
            dtype=image.dtype,
        )
        valid = valid.reshape(-1)
        for idx in range(min(image.shape[0], roi.shape[0], valid.shape[0])):
            if float(valid[idx].item()) <= 0.5:
                continue
            x1, y1, x2, y2 = roi[idx].tolist()
            x1 = max(0, min(image.shape[-1] - 1, int(round(x1))))
            x2 = max(x1 + 1, min(image.shape[-1], int(round(x2))))
            y1 = max(0, min(image.shape[-2] - 1, int(round(y1))))
            y2 = max(y1 + 1, min(image.shape[-2], int(round(y2))))
            mask[idx, :, y1:y2, x1:x2] = 1.0

        batch = dict(batch)
        batch[image_key] = image * mask
        return batch

    def _apply_phase_embedding(self, batch: dict) -> dict:
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if not egexo_cfg or not egexo_cfg.get("use_phase_embedding", True):
            return batch
        if torch is None or self.phase_embedding is None:
            return batch

        state_key = getattr(config, "state_key", "observation.state")
        phase_key = getattr(config, "phase_key", "observation.phase")
        valid_key = getattr(config, "grounding_valid_key", "observation.grounding.valid")
        state = batch.get(state_key)
        phase = batch.get(phase_key)
        if state is None or phase is None:
            return batch
        if not torch.is_tensor(state) or state.ndim != 2:
            return batch
        if not torch.is_tensor(phase):
            return batch

        phase = phase.reshape(state.shape[0], 1).to(device=state.device, dtype=state.dtype)
        phase_embedding = self.phase_embedding(phase) * self.phase_embedding_scale

        valid = batch.get(valid_key)
        if torch.is_tensor(valid):
            valid = valid.reshape(state.shape[0], 1).to(device=state.device, dtype=state.dtype)
            phase_embedding = phase_embedding * valid

        batch = dict(batch)
        batch[state_key] = state + phase_embedding
        return batch

    def _build_phase_gate_input(self, batch: dict):
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if torch is None or self.phase_gate is None:
            return None

        state_key = getattr(config, "state_key", "observation.state")
        ee_pose_key = getattr(config, "ee_pose_key", "observation.ee_pose")
        state = batch.get(state_key)
        if state is None or not torch.is_tensor(state) or state.ndim != 2:
            return None

        features = [state]
        if egexo_cfg.get("phase_gate_use_ee_pose", True):
            ee_pose = batch.get(ee_pose_key)
            if torch.is_tensor(ee_pose) and ee_pose.ndim == 2:
                features.append(ee_pose.to(device=state.device, dtype=state.dtype))
            else:
                zeros = torch.zeros((state.shape[0], 6), device=state.device, dtype=state.dtype)
                features.append(zeros)
        return torch.cat(features, dim=-1)

    def _build_action_head_input(self, batch: dict):
        config = getattr(self, "config", None)
        egexo_cfg = getattr(config, "egexo", {}) if config is not None else {}
        if torch is None:
            return None

        state_key = getattr(config, "state_key", "observation.state")
        ee_pose_key = getattr(config, "ee_pose_key", "observation.ee_pose")
        state = batch.get(state_key)
        if state is None or not torch.is_tensor(state) or state.ndim != 2:
            return None

        features = [state]
        if egexo_cfg.get("dual_action_head_use_ee_pose", True):
            ee_pose = batch.get(ee_pose_key)
            if torch.is_tensor(ee_pose) and ee_pose.ndim == 2:
                features.append(ee_pose.to(device=state.device, dtype=state.dtype))
            else:
                zeros = torch.zeros((state.shape[0], 6), device=state.device, dtype=state.dtype)
                features.append(zeros)
        return torch.cat(features, dim=-1)

    def _compute_phase_head(self, batch: dict):
        if torch is None or self.phase_gate is None:
            return None
        gate_input = self._build_phase_gate_input(batch)
        if gate_input is None:
            return None
        phase_logits = self.phase_gate(gate_input)
        phase_probs = torch.sigmoid(phase_logits)
        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
        }

    def _compute_dual_action_heads(self, batch: dict, phase_head_output):
        if (
            torch is None
            or self.transport_action_head is None
            or self.contact_action_head is None
            or phase_head_output is None
        ):
            return None

        head_input = self._build_action_head_input(batch)
        if head_input is None:
            return None

        transport_action = self.transport_action_head(head_input)
        contact_action = self.contact_action_head(head_input)
        p_contact = phase_head_output["phase_probs"]
        p_transport = 1.0 - p_contact
        mixed_action = p_transport * transport_action + p_contact * contact_action
        return {
            "transport_action": transport_action,
            "contact_action": contact_action,
            "mixed_action": mixed_action,
            "p_transport": p_transport,
            "p_contact": p_contact,
        }

    def _compute_phase_loss(self, batch: dict, phase_head_output):
        config = getattr(self, "config", None)
        if torch is None or phase_head_output is None or config is None:
            return None

        phase_key = getattr(config, "phase_key", "observation.phase")
        valid_key = getattr(config, "grounding_valid_key", "observation.grounding.valid")
        target = batch.get(phase_key)
        if target is None or not torch.is_tensor(target):
            return None

        target = target.reshape(-1, 1).to(
            device=phase_head_output["phase_logits"].device,
            dtype=phase_head_output["phase_logits"].dtype,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            phase_head_output["phase_logits"],
            target,
            reduction="none",
        )

        valid = batch.get(valid_key)
        if torch.is_tensor(valid):
            valid = valid.reshape(-1, 1).to(device=loss.device, dtype=loss.dtype)
            denom = torch.clamp(valid.sum(), min=1.0)
            return (loss * valid).sum() / denom
        return loss.mean()

    def _compute_dual_action_loss(self, batch: dict, dual_action_output):
        if torch is None or dual_action_output is None:
            return None
        target = batch.get("action")
        if target is None or not torch.is_tensor(target):
            return None

        target = target.to(
            device=dual_action_output["mixed_action"].device,
            dtype=dual_action_output["mixed_action"].dtype,
        )
        mixed_loss = torch.nn.functional.smooth_l1_loss(
            dual_action_output["mixed_action"],
            target,
        )
        transport_loss = torch.nn.functional.smooth_l1_loss(
            dual_action_output["transport_action"],
            target,
        )
        contact_loss = torch.nn.functional.smooth_l1_loss(
            dual_action_output["contact_action"],
            target,
        )
        return mixed_loss + 0.5 * (transport_loss + contact_loss)

    def _extract_action_tensor_from_output(self, output):
        if torch is None:
            return None, None
        if torch.is_tensor(output):
            return None, output
        if not isinstance(output, dict):
            return None, None
        for key in ("action", "actions", "pred_action", "pred_actions"):
            value = output.get(key)
            if torch.is_tensor(value):
                return key, value
        return None, None

    def _augment_forward_output(self, output, batch: dict):
        phase_head_output = self._compute_phase_head(batch)
        dual_action_output = self._compute_dual_action_heads(batch, phase_head_output)
        if phase_head_output is None and dual_action_output is None:
            return output

        if isinstance(output, dict):
            output = dict(output)
            config = getattr(self, "config", None)
            loss_cfg = getattr(config, "loss", {}) if config is not None else {}

            if phase_head_output is not None:
                output["phase_logits"] = phase_head_output["phase_logits"]
                output["phase_probs"] = phase_head_output["phase_probs"]

                phase_loss = self._compute_phase_loss(batch, phase_head_output)
                if phase_loss is not None:
                    output["phase_loss"] = phase_loss
                    phase_weight = float(loss_cfg.get("phase_weight", 0.05))
                    if "loss" in output and output["loss"] is not None:
                        output["loss"] = output["loss"] + phase_weight * phase_loss
                    elif "action_loss" in output and output["action_loss"] is not None:
                        output["loss"] = output["action_loss"] + phase_weight * phase_loss
                    else:
                        output["loss"] = phase_weight * phase_loss

            if dual_action_output is not None:
                output["transport_action"] = dual_action_output["transport_action"]
                output["contact_action"] = dual_action_output["contact_action"]
                output["mixed_action"] = dual_action_output["mixed_action"]
                output["p_transport"] = dual_action_output["p_transport"]
                output["p_contact"] = dual_action_output["p_contact"]

                action_key, _ = self._extract_action_tensor_from_output(output)
                if action_key is not None:
                    output[action_key] = dual_action_output["mixed_action"]

                dual_action_loss = self._compute_dual_action_loss(batch, dual_action_output)
                if dual_action_loss is not None:
                    output["dual_action_loss"] = dual_action_loss
                    dual_action_weight = float(loss_cfg.get("dual_action_weight", 0.1))
                    if "loss" in output and output["loss"] is not None:
                        output["loss"] = output["loss"] + dual_action_weight * dual_action_loss
                    elif "action_loss" in output and output["action_loss"] is not None:
                        output["loss"] = output["action_loss"] + dual_action_weight * dual_action_loss
                    else:
                        output["loss"] = dual_action_weight * dual_action_loss
            return output

        return output

    def _prepare_egexo_batch(self, batch: dict) -> dict:
        prepared = copy.copy(batch)
        prepared = self._apply_ego_soft_mask(prepared)
        prepared = self._apply_phase_embedding(prepared)
        return prepared

    def forward(self, batch):  # pragma: no cover - exercised in training env
        prepared = self._prepare_egexo_batch(batch)
        output = super().forward(prepared)
        return self._augment_forward_output(output, prepared)

    def select_action(self, batch):  # pragma: no cover - exercised in inference env
        prepared = self._prepare_egexo_batch(batch)
        output = super().select_action(prepared)
        phase_head_output = self._compute_phase_head(prepared)
        dual_action_output = self._compute_dual_action_heads(prepared, phase_head_output)
        if torch.is_tensor(output) and dual_action_output is not None:
            return dual_action_output["mixed_action"]
        if isinstance(output, dict):
            output = dict(output)
            if phase_head_output is not None:
                output["phase_probs"] = phase_head_output["phase_probs"]
            if dual_action_output is not None:
                output["transport_action"] = dual_action_output["transport_action"]
                output["contact_action"] = dual_action_output["contact_action"]
                output["mixed_action"] = dual_action_output["mixed_action"]
                output["p_transport"] = dual_action_output["p_transport"]
                output["p_contact"] = dual_action_output["p_contact"]
                action_key, _ = self._extract_action_tensor_from_output(output)
                if action_key is not None:
                    output[action_key] = dual_action_output["mixed_action"]
        return output

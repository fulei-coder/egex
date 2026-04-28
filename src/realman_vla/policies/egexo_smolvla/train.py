from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


SMOLVLA_ALLOWED_POLICY_KEYS = {
    "type",
    "repo_id",
    "n_obs_steps",
    "chunk_size",
    "n_action_steps",
    "vlm_model_name",
    "load_vlm_weights",
    "resize_imgs_with_padding",
    "tokenizer_max_length",
    "num_steps",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_state_proj",
    "num_vlm_layers",
    "num_expert_layers",
    "expert_width_multiplier",
    "attention_mode",
    "optimizer_lr",
    "optimizer_betas",
    "optimizer_weight_decay",
    "optimizer_grad_clip_norm",
    "scheduler_warmup_steps",
    "scheduler_decay_steps",
    "scheduler_decay_lr",
}


def _flatten_training_section(cfg: dict) -> dict:
    cfg = dict(cfg or {})
    training = dict(cfg.pop("training", {}) or {})
    for key in (
        "batch_size",
        "steps",
        "save_checkpoint",
        "save_freq",
        "eval_freq",
        "log_freq",
        "output_dir",
        "seed",
        "num_workers",
    ):
        if key in training and key not in cfg:
            cfg[key] = training[key]
    return cfg


def translate_egexo_config(config_path: Path) -> tuple[dict, dict]:
    with config_path.open("r", encoding="utf-8") as f:
        original_cfg = yaml.safe_load(f)
    if not isinstance(original_cfg, dict):
        raise ValueError(f"Invalid egexo config: {config_path}")

    translated_cfg = _flatten_training_section(original_cfg)
    translated_cfg["policy"] = dict(translated_cfg.get("policy", {}) or {})
    policy_cfg = dict(translated_cfg["policy"])
    egexo_meta = {
        "type": policy_cfg.get("type", "egexo_smolvla"),
        "base_policy_type": policy_cfg.get("base_policy_type", "smolvla"),
        "image_keys": policy_cfg.get("image_keys", {}),
        "state_key": policy_cfg.get("state_key", "observation.state"),
        "ee_pose_key": policy_cfg.get("ee_pose_key", "observation.ee_pose"),
        "ego_roi_key": policy_cfg.get("ego_roi_key", "observation.grounding.ego_roi"),
        "grounding_valid_key": policy_cfg.get("grounding_valid_key", "observation.grounding.valid"),
        "phase_key": policy_cfg.get("phase_key", "observation.phase"),
        "egexo": policy_cfg.get("egexo", {}),
        "loss": policy_cfg.get("loss", {}),
        "ablation": translated_cfg.get("ablation", {}),
    }
    policy_cfg["type"] = "smolvla"
    translated_cfg["policy"] = {k: v for k, v in policy_cfg.items() if k in SMOLVLA_ALLOWED_POLICY_KEYS}
    return translated_cfg, egexo_meta


def write_translated_config(translated_cfg: dict, egexo_meta: dict) -> tuple[Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="egexo_smolvla_train_"))
    translated_path = tmp_dir / "smolvla_train_config.yaml"
    meta_path = tmp_dir / "egexo_metadata.yaml"
    with translated_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(translated_cfg, f, sort_keys=False, allow_unicode=True)
    with meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"policy": egexo_meta}, f, sort_keys=False, allow_unicode=True)
    return translated_path, meta_path


def persist_egexo_metadata(meta_path: Path, output_dir: str | Path | None) -> None:
    if not output_dir:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, output_dir / "egexo_metadata.yaml")


def patch_generated_checkpoints(output_dir: str | Path | None, egexo_meta: dict) -> None:
    if not output_dir:
        return
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return

    config_paths = list(checkpoints_dir.glob("*/pretrained_model/config.json"))
    config_paths += list(checkpoints_dir.glob("last/pretrained_model/config.json"))
    seen = set()
    for config_path in config_paths:
        config_path = config_path.resolve()
        if config_path in seen or not config_path.exists():
            continue
        seen.add(config_path)
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.update(egexo_meta)
            cfg["type"] = "egexo_smolvla"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[EgExo] warning: failed to patch {config_path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Train adapter for minimal EgExoSmolVLA stage-A runs")
    parser.add_argument("--config", required=True, help="Path to configs/egexo_smolvla_realman.yaml")
    parser.add_argument("--resume-from", default="", help="Optional checkpoint dir for lerobot resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    translated_cfg, egexo_meta = translate_egexo_config(config_path)
    translated_path, meta_path = write_translated_config(translated_cfg, egexo_meta)
    persist_egexo_metadata(meta_path, translated_cfg.get("output_dir"))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    if args.resume_from:
        ckpt_dir = Path(args.resume_from).resolve()
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={ckpt_dir / 'train_config.json'}",
            "--resume=true",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            "--config",
            str(translated_path),
        ]

    print("[EgExo] translated training config:", translated_path)
    print("[EgExo] egexo metadata:", meta_path)
    print("[EgExo] command:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    patch_generated_checkpoints(translated_cfg.get("output_dir"), egexo_meta)


if __name__ == "__main__":
    main()

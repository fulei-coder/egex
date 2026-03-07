# Third-Party Notices

本项目包含或基于以下第三方开源项目及合作贡献。感谢所有原作者的贡献。

---

## HuggingFace LeRobot

- **来源**: https://github.com/huggingface/lerobot
- **许可证**: Apache License 2.0
- **版权**: Copyright (c) 2024, HuggingFace Inc.

本项目的训练和数据格式转换流程基于 **LeRobot** 框架。
`scripts/convert_to_lerobot.py` 参考了 LeRobot 官方的数据集格式规范和转换逻辑。
`configs/` 下的训练配置参考了 LeRobot 官方示例配置并针对 RM65 硬件进行了适配修改。

LeRobot 本身未被修改或包含在本仓库中，仅作为训练框架依赖。

---

## Vive Tracker / OpenVR

- **来源**: https://github.com/ValveSoftware/openvr
- **参考**: https://github.com/TriadSemi/triad_openvr
- **许可证**: BSD-3-Clause License (OpenVR SDK)

`hardware/vive_tracker.py` 中的 OpenVR 位姿读取逻辑参考了 [triad_openvr](https://github.com/TriadSemi/triad_openvr) 项目的实现思路。
本项目对其进行了重写，适配 RM65 机械臂的遥操作需求。

---

## 合作贡献声明

`scripts/collect_data.py` 数据采集脚本由本项目作者与实验室同门合作完成：
- Vive Tracker 遥操作采集模式的初始代码由合作者提供
- 后续由本项目作者进行重构，增加了示教模式、双相机支持、数据格式化等功能

---

## 算法实现参考

以下算法的模型实现均来自 LeRobot 框架内置或社区贡献的实现，本项目未重新实现模型本身：

| 算法 | 原始论文 | LeRobot 内实现来源 |
|------|---------|-------------------|
| ACT | [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) | `lerobot.common.policies.act` |
| Diffusion Policy | [Chi et al., 2023](https://arxiv.org/abs/2303.04137) | `lerobot.common.policies.diffusion` |
| VQ-BeT | [Lee et al., 2024](https://arxiv.org/abs/2403.03181) | `lerobot.common.policies.vqbet` |
| SmolVLA | [HuggingFace, 2024](https://huggingface.co/lerobot/smolvla_base) | `lerobot.common.policies.smolvla` |
| Pi0 | [Black et al., 2024](https://www.physicalintelligence.company/blog/pi0) | `lerobot.common.policies.pi0` |

---

## 睿尔曼 RM_API2

- **来源**: https://github.com/RealManRobot/RM_API2
- **许可证**: Apache License 2.0
- **版权**: Copyright 2024 realman-robotics

本项目通过 `Robotic_Arm` Python SDK 与 RM65 机械臂通信，该 SDK 作为 pip 依赖引用。

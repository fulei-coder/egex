# 🔌 硬件配置指南

## 机械臂 — 睿尔曼 RM65-B

### 连接配置
- **通信协议**: TCP/IP
- **默认IP**: `192.168.2.18`（出厂值，可通过示教器修改）
- **默认端口**: `8080`
- **SDK**: [RM_API2 Python](https://www.realman-robotics.cn/)

### 初始化流程
```python
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("<YOUR_ARM_IP>", 8080)

# 基础设置
arm.rm_set_arm_run_mode(1)     # 设置运行模式
arm.rm_set_tool_voltage(3)     # 工具端电压 3V
arm.rm_set_modbus_mode(1, 9600, 2)  # Modbus: 端口1, 9600波特率
```

### 运动控制
| API | 说明 | 阻塞 |
|-----|------|------|
| `rm_movej(joints, speed, 0, 0, 1)` | 关节空间运动 | 阻塞 |
| `rm_movej(joints, speed, 0, 0, 0)` | 关节空间运动 | 非阻塞 |
| `rm_movep_canfd(pose, False, 0, 60)` | 笛卡尔空间CANFD | 非阻塞 |
| `rm_get_current_arm_state()` | 获取当前状态 | - |

---

## 夹爪 — FAE2M86C (Modbus RTU)

### Modbus 参数
| 参数 | 值 |
|------|---|
| 端口 (port) | 1 |
| 寄存器地址 (address) | 43 |
| 设备号 (device) | 1 |
| 寄存器数量 (num) | 2 |
| 波特率 | 9600 |
| 停止位 | 2 |

### 编码方式
夹爪位置通过 4 字节寄存器值编码：
```python
def dec_to_register(dec):
    """小数 (0=全开, 1=全闭) → 4字节寄存器"""
    value = dec * 256000
    R0 = int(value // (256**3))
    R1 = int((value % (256**3)) // (256**2))
    R2 = int((value % (256**2)) // 256)
    R3 = int(value % 256)
    return [R0, R1, R2, R3]

# 全开: [0, 0, 0, 0]
# 全闭: [0, 3, 232, 0]
```

### ⚠️ Modbus 超时陷阱

Modbus 读写单次可能耗时 **300-500ms**（9600 波特率下的超时机制），同步执行会将主控制循环频率从 30Hz 拖到 1-2Hz。

> 容易误判为模型推理瓶颈。加 timing 日志可快速定位。

**解决方案**：异步写入 + 缓存读取
```python
# ❌ 同步阻塞（不要这样做）
gripper_pos = arm.rm_read_multiple_holding_registers(params)  # 可能500ms

# ✅ 异步写入 + 缓存状态
threading.Thread(target=write_gripper, daemon=True).start()
# 夹爪只有开/闭两种状态，不需要实时读取
```

---

## 相机 — Intel RealSense D435i

### 双相机配置
| 位置 | 序列号 | 用途 |
|------|--------|------|
| 顶部 | `<YOUR_TOP_CAM_SN>` | 全局视角 |
| 腕部 | `<YOUR_WRIST_CAM_SN>` | 精细视角 |

### 推荐参数
| 参数 | 值 | 说明 |
|------|---|------|
| 分辨率 | 640×480 | 平衡质量与速度 |
| 帧率 | 30fps | 与采集频率对齐 |
| 自动曝光 | **关闭** | 自动曝光会导致训练/推理图像不一致 |
| 手动曝光 | 150 | 根据环境光调整，室内日光灯环境参考值 |

### 相机占用问题

如果出现 "Device is already in use"——通常是上次脚本没正常退出，RealSense 设备锁没释放：
```bash
# 找到并杀掉占用进程
pkill -f realsense
pkill -f collect_data

# 实在不行就物理拔插 USB
```

另外两个相机**不要接同一个 USB 控制器**，带宽不够也会报这个错。

---

## 遥操作 — HTC Vive Tracker 3.0

### 前置要求
1. 安装 [SteamVR](https://store.steampowered.com/app/250820/SteamVR/)
2. 安装 OpenVR Python: `pip install openvr`
3. 配对 Vive Tracker

### SteamVR 启停
```bash
# 启动
~/.steam/debian-installation/ubuntu12_64/steam-runtime/run.sh \
    ~/.local/share/Steam/steamapps/common/SteamVR/bin/vrserver &

# 停止
pkill -f vrserver
```

### 坐标映射
Vive Tracker 坐标系与机械臂坐标系不一致，需要映射：
```
Vive → Robot:
  Robot_X = -Vive_Z × scale
  Robot_Y = -Vive_X × scale
  Robot_Z = +Vive_Y × scale
```

### 校准流程
1. 将 Tracker 放在固定位置
2. 按 `v` 校准（记录零点）
3. 按 `w` 启用遥控
4. 移动 Tracker 控制机械臂

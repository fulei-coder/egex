# 🔧 System Setup

**⚠️ 重要声明：当前采集只使用 ROS2 流程，不使用 ROS1/Noetic/catkin/rosrun。**

## Prerequisites

- **Ubuntu 22.04** (Recommended for ROS2 Humble)
- [**ROS2 Humble**](https://docs.ros.org/en/humble/Installation.html)
- **Python 3.9+**

---

## Step-by-Step Setup

1. **Install Python Dependencies**

   ```sh
   # install required packages
   pip install -r requirements.txt
   pip install -r ros2_requirements.txt
   ```

2. **Build and Source Environment**

   Instead of `catkin_make`, use `colcon build` for ROS2:

   ```bash
   source /opt/ros/humble/setup.bash
   # Make sure you are in the workspace root that contains src/uarm
   colcon build
   source install/setup.bash
   ```

   *If you do not have a full colcon workspace setup, you can still run the script directly using Python 3:*
   ```bash
   source /opt/ros/humble/setup.bash
   python3 src/uarm/scripts/Uarm_teleop/servo_reader.py --ros-args -p serial_port:=/dev/ttyUSB0
   ```

---

# 🤖 Plug-and-Play with Real Robot with ROS2

## 1. Verify Teleop Arm Output & Publish Data

In a new terminal, check servo readings and publish to the ROS2 topic (No `roscore` needed in ROS2):

```sh
source /opt/ros/humble/setup.bash
conda activate lerobot
python3 src/uarm/scripts/Uarm_teleop/servo_reader.py \
  --ros-args -p serial_port:=/dev/ttyUSB0
```

*Replace `/dev/ttyUSB0` with your actual serial port if different.*

Your teleop arm now publishes to the `/servo_angles` topic. You can verify it by opening another terminal and running:
```sh
source /opt/ros/humble/setup.bash
ros2 topic echo /servo_angles
```

## 2. Control the Follower Arm (RealMan)

Return to the main repository root and run the data collection or inference script as defined in the main project summaries, which will subscribe directly to the `/servo_angles` topic published by `servo_reader.py`.

---

# 🖥️ Try It Out in Simulation

If you do not have robot hardware, you can try teleoperation in simulation.  
See detailed guidance [here](https://github.com/MINT-SJTU/Lerobot-Anything-U-arm/blob/feat/simulation/src/simulation/README.md).

# U-arm Teleop Scripts

**⚠️ 重要声明 / Important:**
- `servo_reader.py`: **推荐入口 / Recommended Entrypoint**. Reads all 7 servos for full 6DoF+1DoF gripper teleoperation. Used exclusively in the data collection and inference pipeline. Accepts `--ros-args -p serial_port:=/dev/ttyUSBX`.
- `servo_reader_fixed.py`: **只读局部调试脚本 / Deprecated/Partial Debugging Script**. ONLY reads 3 servos (0, 1, 2) and keeps the rest at 0.0. NOT FOR USE IN FULL DATA COLLECTION. Do not use unless you are specifically debugging the base joints.

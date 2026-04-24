import time
import numpy as np
import gymnasium as gym
import mani_skill.envs  # 必须导入以注册所有 env/agent

def main():
    env = gym.make(
        "PushCube-v1", # "Empty-v1",
        robot_uids="xarm6_robotiq", 
        render_mode="human",        # 可换 "human" 以窗口渲染
        control_mode="pd_joint_pos",
    )
    obs, _ = env.reset(seed=0)
    print("Action space:", env.action_space)

    # 两组关节角（弧度制示例），可自行调整：
    pose_a = np.radians([14.1, -8, -24.7, 196.9, 62.3, -8.8, 0.0])
    pose_b = np.radians([-30, -8, 0, 196.9, 62.3, -8.8, 0.0])

    steps = 200            # 一次往返的步数
    dwell = 0.01           # 每步停留时间（秒）

    while True:
        # 从 pose_a 逐步过渡到 pose_b
        for t in np.linspace(0, 1, steps):
            action = (1 - t) * pose_a + t * pose_b
            env.step(action)
            env.render()   # 如果为 "sensors"，此处可取返回图像做处理
            time.sleep(dwell)

        # 再从 pose_b 逐步回到 pose_a
        for t in np.linspace(0, 1, steps):
            action = (1 - t) * pose_b + t * pose_a
            env.step(action)
            env.render()
            time.sleep(dwell)

    env.close()

if __name__ == "__main__":
    main()

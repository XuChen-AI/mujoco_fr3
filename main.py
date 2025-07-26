import mujoco
import mujoco.viewer

# 从XML文件加载MuJoCo模型 - 使用包含场景背景的文件
m = mujoco.MjModel.from_xml_path('description/scene.xml')
d = mujoco.MjData(m)

# 设置机械臂到常规初始位置（home姿态）
# FR3的常规位置：关节角度设置为适合操作的姿态
home_position = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # 弧度制
d.qpos[:7] = home_position  # 设置关节位置
d.ctrl[:7] = home_position  # 设置控制器目标

# 前向运动学计算，更新机械臂状态
mujoco.mj_forward(m, d)

# 启动交互式可视化器
with mujoco.viewer.launch_passive(m, d) as viewer:
    # 开始仿真循环
    while viewer.is_running():
        # 执行一个仿真步骤
        mujoco.mj_step(m, d)
        
        # 更新可视化器
        viewer.sync()
        
        # 重置条件：如果仿真时间超过10秒，重置仿真
        if d.time > 10.0:
            mujoco.mj_resetData(m, d)
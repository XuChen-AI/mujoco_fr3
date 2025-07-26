import mujoco
import mujoco.viewer

# 从XML文件加载MuJoCo模型
m = mujoco.MjModel.from_xml_path('description/fr3.xml')
d = mujoco.MjData(m)

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
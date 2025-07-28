#!/usr/bin/env python3
"""
增强版Mujoco仿真示例

包含轨迹规划测试功能：
1. 加载带坐标轴的FR3机器人模型
2. 设置home姿态为起始位置
3. 接受用户输入的目标点坐标
4. 计算到目标点的逆运动学解
5. 生成轨迹并执行动画
6. 使用运动学、规划器等模块
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import time
import threading
from queue import Queue, Empty

# 添加项目根目录到Python路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics
from planning.trajectory_planner import TrajectoryPlanner


class FR3TrajectoryDemo:
    """FR3机械臂轨迹规划演示类"""
    
    def __init__(self):
        """初始化演示环境"""
        # 获取模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, '..', 'description', 'scene_with_axes.xml')
        
        # 加载MuJoCo模型和数据
        print("正在加载FR3机器人模型...")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化运动学和规划器
        self.fk = ForwardKinematics(self.model, self.data)
        self.ik = InverseKinematics(self.model, self.data)
        self.planner = TrajectoryPlanner()
        
        # FR3标准home姿态的关节角度 (参考MuJoCo模型中的keyframe定义)
        self.home_position = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        
        # 设置初始位置
        self.data.qpos[:7] = self.home_position
        # 同时设置控制器目标位置为home位置
        self.data.ctrl[:7] = self.home_position
        # 初始化保持位置为home位置
        self.hold_position_joints = self.home_position.copy()
        mujoco.mj_forward(self.model, self.data)
        
        # 计算home位置下的末端执行器位置
        self.home_ee_position, _ = self.fk.compute_fk(self.home_position)
        print(f"Home位置下的末端执行器坐标: [{self.home_ee_position[0]:.3f}, {self.home_ee_position[1]:.3f}, {self.home_ee_position[2]:.3f}]")
        
        # 轨迹相关变量
        self.trajectory = None
        self.trajectory_index = 0
        self.executing_trajectory = False
        self.trajectory_speed = 1  # 控制轨迹执行速度的因子
        self.trajectory_completed = False  # 标记轨迹是否已完成
        self.waiting_for_input = True  # 标记是否等待用户输入
        self.current_target_position = None  # 当前目标位置
        self.hold_position_joints = None  # 需要保持的关节角度
        
        # 用于处理用户输入的队列和线程
        self.input_queue = Queue()
        self.input_thread = None
        self.should_exit = False
    
    def input_thread_function(self):
        """在单独线程中处理用户输入"""
        while not self.should_exit:
            try:
                if self.waiting_for_input:
                    target_position = self.get_user_target_interactive()
                    if target_position is None:
                        # 用户选择退出
                        self.input_queue.put(('exit', None))
                        break
                    else:
                        # 将目标位置放入队列
                        self.input_queue.put(('target', target_position))
                        # 设置为不等待输入，避免重复获取
                        self.waiting_for_input = False
                else:
                    # 如果不需要输入，等待较长时间减少CPU占用
                    time.sleep(0.5)
            except Exception as e:
                print(f"输入线程错误: {e}")
                break
    def get_user_target_interactive(self):
        """获取用户输入的目标位置坐标"""
        print("\n" + "="*50)
        print("请输入目标位置坐标 (单位: 米)")
        current_ee_pos, _ = self.fk.compute_fk(self.data.qpos[:7])
        print(f"当前位置: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
        print("输入 'q' 退出程序")
        print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                x_input = input("X坐标: ").strip()
                if x_input.lower() in ['q', 'quit']:
                    return None
                x = float(x_input)
                
                y_input = input("Y坐标: ").strip()
                if y_input.lower() in ['q', 'quit']:
                    return None
                y = float(y_input)
                
                z_input = input("Z坐标: ").strip()
                if z_input.lower() in ['q', 'quit']:
                    return None
                z = float(z_input)
                
                target_position = np.array([x, y, z])
                
                # 验证目标位置的合理性
                if self._validate_target_position(target_position):
                    print(f"目标设定: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    return target_position
                else:
                    print("目标位置超出工作范围，请重新输入")
                    
            except ValueError:
                print("输入格式错误，请输入数字")
            except KeyboardInterrupt:
                print("\n用户取消输入")
                return None
    
    def _validate_target_position(self, target_position):
        """验证目标位置是否在合理范围内"""
        # 简单的范围检查：距离base不超过1.5米
        distance_from_base = np.linalg.norm(target_position)
        if distance_from_base > 1.5:
            print(f"目标距离base过远: {distance_from_base:.3f}m > 1.5m")
            return False
            
        # 检查Z坐标不能太低（避免撞击地面）
        if target_position[2] < 0.1:
            print(f"Z坐标过低: {target_position[2]:.3f}m < 0.1m")
            return False
            
        return True
    
    def plan_trajectory_to_target(self, target_position):
        """规划从当前位置到目标位置的轨迹"""
        print(f"规划轨迹到 [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]...")
        
        # 获取当前关节角度作为起始点
        current_joint_angles = self.data.qpos[:7].copy()
        
        # 使用逆运动学求解目标关节角度
        target_joint_angles, success, error = self.ik.solve_ik_position_only(
            target_position, 
            initial_guess=current_joint_angles,  # 使用当前位置作为初始猜测
            max_iterations=200,
            tolerance=1e-3
        )
        
        if not success:
            print(f"❌ 逆运动学求解失败，误差: {error:.6f}")
            print("目标位置无法到达，请重新输入")
            return False
        
        # 使用轨迹规划器的智能规划功能：考虑关节运动的舒适性和自然性
        optimized_trajectory = self.planner.plan_comfortable_trajectory(
            current_joint_angles, 
            target_joint_angles
        )
        
        self.trajectory = optimized_trajectory
        print(f"✓ 轨迹规划完成，开始执行...")
        
        # 重置轨迹执行状态
        self.trajectory_index = 0
        self.executing_trajectory = True
        self.trajectory_completed = False
        self.current_target_position = target_position.copy()  # 保存目标位置用于验证
        # 清除保持位置，因为开始新的轨迹执行
        self.hold_position_joints = None
        
        return True
    
    def execute_trajectory_step(self):
        """执行轨迹的一步"""
        if not self.executing_trajectory or self.trajectory is None:
            return False
            
        if self.trajectory_index >= len(self.trajectory):
            # 轨迹执行完毕
            self.executing_trajectory = False
            self.trajectory_completed = True
            
            # 保持在最终位置：设置控制器目标为最终关节角度
            final_joint_angles = self.trajectory[-1]
            self.data.ctrl[:7] = final_joint_angles
            # 更新需要保持的关节角度，确保在等待输入期间刚性保持位置
            self.hold_position_joints = final_joint_angles.copy()
            
            # 验证是否到达目标位置
            self.verify_target_reached()
            
            # 准备接受下一次用户输入
            self.waiting_for_input = True
            return False
        
        # 设置当前轨迹点的关节角度
        current_joint_angles = self.trajectory[self.trajectory_index]
        self.data.qpos[:7] = current_joint_angles
        # 同时设置控制器目标位置，确保控制器不会拉回到之前的位置
        self.data.ctrl[:7] = current_joint_angles
        
        # 更新轨迹索引（控制执行速度）
        self.trajectory_index += self.trajectory_speed
        
        return True
    
    def maintain_position(self):
        """在等待用户输入期间保持当前位置（刚性保持）"""
        if self.hold_position_joints is not None and not self.executing_trajectory:
            # 持续设置控制器目标位置，确保机械臂刚性保持在目标位置
            # 这是关键：每次仿真步都要重新设置控制目标，防止位置漂移
            self.data.ctrl[:7] = self.hold_position_joints
    
    def verify_target_reached(self):
        """验证机械臂是否到达了目标位置并输出验证结果"""
        if self.current_target_position is None:
            return
            
        # 计算当前末端执行器位置
        current_ee_pos, _ = self.fk.compute_fk(self.data.qpos[:7])
        
        # 计算位置误差
        position_error = np.linalg.norm(current_ee_pos - self.current_target_position)
        
        print(f"\n" + "="*50)
        print("✓ 轨迹执行完成！")
        print(f"目标位置: [{self.current_target_position[0]:.3f}, {self.current_target_position[1]:.3f}, {self.current_target_position[2]:.3f}]")
        print(f"实际位置: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
        print(f"位置误差: {position_error:.3f} mm")
        
        # 判断是否成功到达
        tolerance = 0.01  # 1cm容差
        if position_error <= tolerance:
            print("🎯 成功到达目标位置！")
        else:
            print("⚠️  未完全到达目标位置")
        
        print("="*50)

    
    def run_interactive_demo(self):
        """运行交互式轨迹规划演示"""
        print("=== FR3机械臂交互式控制 ===")
        print("使用说明：")
        print("1. 仿真界面启动，机械臂处于标准home位置")
        print("2. 在终端输入目标坐标(x, y, z)")
        print("3. 机械臂执行轨迹到目标位置")
        print("4. 到达后验证位置并等待下一个目标")
        print("5. 输入 'q' 退出程序")
        print("=" * 40)
        
        # 显示初始状态
        print(f"Home位置: [{self.home_ee_position[0]:.3f}, {self.home_ee_position[1]:.3f}, {self.home_ee_position[2]:.3f}]")
        print("启动仿真界面...")
        
        # 启动用户输入线程
        self.input_thread = threading.Thread(target=self.input_thread_function, daemon=True)
        self.input_thread.start()
        
        print("🚀 仿真已启动！请在终端输入目标坐标。")
        
        # 主仿真循环
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                
                # 检查用户输入队列
                try:
                    command, data = self.input_queue.get_nowait()
                    if command == 'exit':
                        print("退出程序...")
                        break
                    elif command == 'target':
                        # 收到新的目标位置，规划轨迹
                        if self.plan_trajectory_to_target(data):
                            pass  # 轨迹规划成功，开始执行
                        else:
                            print("轨迹规划失败，请重新输入")
                            self.waiting_for_input = True
                except Empty:
                    # 队列为空，继续正常执行
                    pass
                
                # 如果正在执行轨迹
                if self.executing_trajectory:
                    self.execute_trajectory_step()
                else:
                    # 如果不在执行轨迹，保持当前位置（刚性保持）
                    self.maintain_position()
                
                # 更新仿真（关键：确保位置控制器正常工作）
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 控制仿真速度
                time.sleep(0.01)  # 10ms延迟，使动画平滑
        
        # 清理资源
        self.should_exit = True
        if self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)


def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = FR3TrajectoryDemo()
        
        # 运行交互式演示
        demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

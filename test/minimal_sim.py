#!/usr/bin/env python3
"""
增强版Mujoco仿真示例 - 简化版本

包含轨迹规划测试功能：
1. 加载带坐标轴的FR3机器人模型
2. 设置home姿态为起始位置
3. 接受用户输入的目标点坐标
4. 计算到目标点的逆运动学解
5. 生成轨迹并执行动画
6. 使用运动学、规划器等模块

注意：为避免多线程导致的段错误，此版本使用简化的交互方式
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import time

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
        self.current_target_position = None  # 当前目标位置
        self.hold_position_joints = None  # 需要保持的关节角度
        
        # 用户交互状态
        self.waiting_for_input = True  # 是否等待用户输入
        self.demo_mode = False  # 是否为演示模式
        
        # 简化的目标位置列表（预定义一些测试位置）
        self.test_targets = [
            [0.6, 0.2, 0.4],   # 右前方
            [0.6, -0.2, 0.4],  # 右后方
            [0.4, 0.0, 0.3],   # 正前方低位
            [0.4, 0.0, 0.7],   # 正前方高位
        ]
        self.current_target_index = 0
    
    def get_user_input_interactive(self):
        """获取用户输入的目标位置坐标（安全的单线程版本）"""
        print("\n" + "="*50)
        print("请输入目标位置坐标 (单位: 米)")
        current_ee_pos, _ = self.fk.compute_fk(self.data.qpos[:7])
        print(f"当前位置: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
        print("输入 'demo' 进入自动演示模式")
        print("输入 'q' 退出程序")
        print("-" * 50)
        
        try:
            # 获取用户输入
            x_input = input("X坐标: ").strip()
            if x_input.lower() in ['q', 'quit']:
                return None
            if x_input.lower() == 'demo':
                return 'demo'
            x = float(x_input)
            
            y_input = input("Y坐标: ").strip()
            if y_input.lower() in ['q', 'quit']:
                return None
            if y_input.lower() == 'demo':
                return 'demo'
            y = float(y_input)
            
            z_input = input("Z坐标: ").strip()
            if z_input.lower() in ['q', 'quit']:
                return None
            if z_input.lower() == 'demo':
                return 'demo'
            z = float(z_input)
            
            target_position = np.array([x, y, z])
            
            # 验证目标位置的合理性
            if self._validate_target_position(target_position):
                print(f"目标设定: [{x:.3f}, {y:.3f}, {z:.3f}]")
                return target_position
            else:
                print("目标位置超出工作范围，请重新输入")
                return self.get_user_input_interactive()  # 递归重新获取输入
                
        except ValueError:
            print("输入格式错误，请输入数字")
            return self.get_user_input_interactive()  # 递归重新获取输入
        except KeyboardInterrupt:
            print("\n用户取消输入")
            return None
    
    def get_next_target(self):
        """获取下一个目标位置（演示模式）"""
        if self.current_target_index < len(self.test_targets):
            target = np.array(self.test_targets[self.current_target_index])
            self.current_target_index += 1
            return target
        else:
            # 回到home位置
            return self.home_ee_position.copy()
    
    def handle_user_interaction(self):
        """处理用户交互（在轨迹完成后调用）"""
        if not self.demo_mode:
            # 手动输入模式
            target_input = self.get_user_input_interactive()
            
            if target_input is None:
                return False  # 用户选择退出
            elif isinstance(target_input, str) and target_input == 'demo':
                # 切换到演示模式
                self.demo_mode = True
                self.current_target_index = 0
                print("🎭 切换到自动演示模式")
                return True
            else:
                # 用户输入了目标位置
                if self.plan_trajectory_to_target(target_input):
                    self.waiting_for_input = False
                    return True
                else:
                    print("轨迹规划失败，请重新输入")
                    return True  # 继续等待输入
        else:
            # 演示模式
            if self.current_target_index < len(self.test_targets):
                next_target = self.get_next_target()
                print(f"\n🎯 演示模式：移动到目标 {self.current_target_index}: [{next_target[0]:.3f}, {next_target[1]:.3f}, {next_target[2]:.3f}]")
                if self.plan_trajectory_to_target(next_target):
                    self.waiting_for_input = False
                    return True
            else:
                # 演示完成，询问是否继续
                print("\n🎭 演示完成！")
                choice = input("输入 'c' 继续演示，'m' 切换到手动模式，或 'q' 退出: ").strip().lower()
                if choice == 'c':
                    self.current_target_index = 0  # 重置演示
                    return True
                elif choice == 'm':
                    self.demo_mode = False
                    print("🔧 切换到手动输入模式")
                    return True
                else:
                    return False  # 退出
        
        return True
    
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
            
            # 设置等待用户输入
            self.waiting_for_input = True
            self.trajectory_completed = True
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
        print("2. 在终端输入目标坐标(x, y, z)或输入'demo'进入自动演示")
        print("3. 机械臂执行轨迹到目标位置")
        print("4. 到达后验证位置并等待下一个目标")
        print("5. 输入 'q' 退出程序")
        print("=" * 40)
        
        # 显示初始状态
        print(f"Home位置: [{self.home_ee_position[0]:.3f}, {self.home_ee_position[1]:.3f}, {self.home_ee_position[2]:.3f}]")
        print("启动仿真界面...")
        
        print("🚀 仿真已启动！")
        
        # 主仿真循环
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 初始输入提示
            initial_input_shown = False
            
            while viewer.is_running():
                
                # 显示初始输入提示
                if self.waiting_for_input and not initial_input_shown:
                    print("\n📝 请在仿真运行时，在终端输入目标坐标或命令")
                    initial_input_shown = True
                
                # 处理用户交互（当需要输入且不在执行轨迹时）
                if self.waiting_for_input and not self.executing_trajectory:
                    if not self.handle_user_interaction():
                        print("退出程序...")
                        break
                
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

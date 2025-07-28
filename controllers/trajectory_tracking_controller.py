"""
轨迹跟踪控制器模块
Trajectory Tracking Controller for FR3 Robot Arm

实现PD控制器用于跟踪预定义的关节空间轨迹
"""

import numpy as np
import mujoco


class TrajectoryTrackingController:
    """
    轨迹跟踪控制器类
    
    使用PD控制实现对关节空间轨迹的精确跟踪。
    支持可调节的比例增益(Kp)和微分增益(Kd)参数。
    """
    
    def __init__(self, 
                 kp=np.array([1000, 1000, 1000, 1000, 500, 500, 500]),  # 比例增益
                 kd=np.array([50, 50, 50, 50, 25, 25, 25]),            # 微分增益
                 max_torque=np.array([87, 87, 87, 87, 12, 12, 12])):   # 最大力矩限制
        """
        初始化轨迹跟踪控制器
        
        Args:
            kp (np.ndarray): 比例增益数组，长度为7 (对应7个关节)
            kd (np.ndarray): 微分增益数组，长度为7 (对应7个关节)
            max_torque (np.ndarray): 最大力矩限制数组，长度为7
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.max_torque = np.array(max_torque)
        
        # 轨迹相关变量
        self.trajectory = None          # 当前轨迹 [N x 7] 数组
        self.trajectory_time = None     # 轨迹时间戳数组
        self.current_step = 0           # 当前轨迹步数
        self.trajectory_started = False # 轨迹是否已开始
        self.start_time = 0.0          # 轨迹开始时间
        
        # 记录变量用于性能分析
        self.position_errors = []       # 位置误差历史
        self.velocity_errors = []       # 速度误差历史
        self.control_torques = []       # 控制力矩历史
        
    def set_trajectory(self, trajectory, dt=0.002):
        """
        设置要跟踪的轨迹
        
        Args:
            trajectory (np.ndarray): 关节角度轨迹，shape为[N, 7]
            dt (float): 时间步长，默认0.002秒
        """
        self.trajectory = np.array(trajectory)
        self.trajectory_time = np.arange(len(trajectory)) * dt
        self.current_step = 0
        self.trajectory_started = False
        
        # 清空历史记录
        self.position_errors = []
        self.velocity_errors = []
        self.control_torques = []
        
        print(f"轨迹已设置: {len(trajectory)}个点, 总时长: {self.trajectory_time[-1]:.2f}秒")
    
    def start_trajectory(self, current_time):
        """
        开始轨迹跟踪
        
        Args:
            current_time (float): 当前仿真时间
        """
        self.trajectory_started = True
        self.start_time = current_time
        self.current_step = 0
        print(f"轨迹跟踪开始，起始时间: {current_time:.2f}秒")
    
    def get_desired_state(self, current_time):
        """
        根据当前时间获取期望的关节状态
        
        Args:
            current_time (float): 当前仿真时间
            
        Returns:
            tuple: (desired_positions, desired_velocities)
                - desired_positions: 期望关节角度 [7,]
                - desired_velocities: 期望关节速度 [7,]
        """
        if not self.trajectory_started or self.trajectory is None:
            return np.zeros(7), np.zeros(7)
        
        # 计算相对时间
        relative_time = current_time - self.start_time
        
        # 如果轨迹结束，保持最后一个点
        if relative_time >= self.trajectory_time[-1]:
            return self.trajectory[-1], np.zeros(7)
        
        # 找到当前时间对应的轨迹索引
        step_idx = np.searchsorted(self.trajectory_time, relative_time)
        step_idx = min(step_idx, len(self.trajectory) - 1)
        
        desired_positions = self.trajectory[step_idx]
        
        # 计算期望速度（数值微分）
        if step_idx < len(self.trajectory) - 1:
            dt = self.trajectory_time[step_idx + 1] - self.trajectory_time[step_idx]
            desired_velocities = (self.trajectory[step_idx + 1] - self.trajectory[step_idx]) / dt
        else:
            desired_velocities = np.zeros(7)
        
        self.current_step = step_idx
        return desired_positions, desired_velocities
    
    def compute_control(self, model, data, current_time, gravity_compensation=True):
        """
        计算控制力矩
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            current_time (float): 当前仿真时间
            gravity_compensation (bool): 是否启用重力补偿
            
        Returns:
            np.ndarray: 控制力矩数组 [7,]
        """
        # 获取期望状态
        q_desired, qd_desired = self.get_desired_state(current_time)
        
        # 获取当前状态
        q_current = data.qpos[:7].copy()
        qd_current = data.qvel[:7].copy()
        
        # 计算误差
        position_error = q_desired - q_current
        velocity_error = qd_desired - qd_current
        
        # PD控制
        tau_pd = self.kp * position_error + self.kd * velocity_error
        
        # 重力补偿
        tau_gravity = np.zeros(7)
        if gravity_compensation:
            # 使用MuJoCo计算重力补偿力矩
            tau_gravity = data.qfrc_bias[:7].copy()
        
        # 总控制力矩
        tau_total = tau_pd + tau_gravity
        
        # 力矩限制
        tau_clamped = np.clip(tau_total, -self.max_torque, self.max_torque)
        
        # 记录数据用于分析
        self.position_errors.append(position_error.copy())
        self.velocity_errors.append(velocity_error.copy())
        self.control_torques.append(tau_clamped.copy())
        
        return tau_clamped
    
    def is_trajectory_complete(self, current_time):
        """
        检查轨迹是否已完成
        
        Args:
            current_time (float): 当前仿真时间
            
        Returns:
            bool: 轨迹是否完成
        """
        if not self.trajectory_started or self.trajectory is None:
            return False
        
        relative_time = current_time - self.start_time
        return relative_time >= self.trajectory_time[-1]
    
    def get_tracking_performance(self):
        """
        获取轨迹跟踪性能指标
        
        Returns:
            dict: 包含各种性能指标的字典
        """
        if not self.position_errors:
            return {}
        
        position_errors = np.array(self.position_errors)
        velocity_errors = np.array(self.velocity_errors)
        
        # 计算RMS误差
        rms_position_error = np.sqrt(np.mean(position_errors**2, axis=0))
        rms_velocity_error = np.sqrt(np.mean(velocity_errors**2, axis=0))
        
        # 计算最大误差
        max_position_error = np.max(np.abs(position_errors), axis=0)
        max_velocity_error = np.max(np.abs(velocity_errors), axis=0)
        
        # 计算总体误差指标
        total_rms_position = np.sqrt(np.mean(rms_position_error**2))
        total_max_position = np.max(max_position_error)
        
        return {
            'rms_position_error_per_joint': rms_position_error,
            'rms_velocity_error_per_joint': rms_velocity_error,
            'max_position_error_per_joint': max_position_error,
            'max_velocity_error_per_joint': max_velocity_error,
            'total_rms_position_error': total_rms_position,
            'total_max_position_error': total_max_position,
            'num_control_steps': len(self.position_errors)
        }
    
    def reset(self):
        """
        重置控制器状态
        """
        self.trajectory = None
        self.trajectory_time = None
        self.current_step = 0
        self.trajectory_started = False
        self.start_time = 0.0
        
        # 清空历史记录
        self.position_errors = []
        self.velocity_errors = []
        self.control_torques = []
        
        print("轨迹跟踪控制器已重置")
    
    def update_gains(self, kp=None, kd=None):
        """
        更新控制增益
        
        Args:
            kp (np.ndarray): 新的比例增益，可选
            kd (np.ndarray): 新的微分增益，可选
        """
        if kp is not None:
            self.kp = np.array(kp)
            print(f"比例增益已更新: {self.kp}")
        
        if kd is not None:
            self.kd = np.array(kd)
            print(f"微分增益已更新: {self.kd}")
    
    def print_current_status(self, current_time):
        """
        打印当前跟踪状态信息
        
        Args:
            current_time (float): 当前仿真时间
        """
        if not self.trajectory_started:
            print("轨迹跟踪尚未开始")
            return
        
        relative_time = current_time - self.start_time
        progress = min(relative_time / self.trajectory_time[-1] * 100, 100)
        
        print(f"轨迹进度: {progress:.1f}% | "
              f"步数: {self.current_step}/{len(self.trajectory)} | "
              f"时间: {relative_time:.2f}/{self.trajectory_time[-1]:.2f}s")
        
        # 显示最近的误差
        if self.position_errors:
            latest_error = self.position_errors[-1]
            max_error = np.max(np.abs(latest_error))
            print(f"当前最大位置误差: {max_error:.4f} rad ({np.degrees(max_error):.2f} deg)")

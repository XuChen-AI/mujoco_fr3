"""
导纳控制器实现
Admittance Controller for FR3 Robot Arm

导纳控制是一种力控制方法，通过调节机械臂的位置/速度来响应外部力，
使机械臂表现出指定的动态特性（质量、阻尼、刚度）。
"""

import numpy as np
import mujoco


class AdmittanceController:
    """
    导纳控制器类
    
    实现基于力反馈的导纳控制，使机械臂在接触时表现出柔顺性。
    控制方程: M_d * ddx + B_d * dx + K_d * (x - x_d) = F_ext
    
    其中:
    - M_d: 期望惯性矩阵
    - B_d: 期望阻尼矩阵  
    - K_d: 期望刚度矩阵
    - F_ext: 外部力
    - x_d: 期望位置
    """
    
    def __init__(self, 
                 desired_mass=1.0,      # 期望质量 (kg)
                 desired_damping=20.0,   # 期望阻尼 (N*s/m)
                 desired_stiffness=100.0, # 期望刚度 (N/m)
                 dt=0.002):             # 时间步长 (s)
        """
        初始化导纳控制器
        
        Args:
            desired_mass: 期望的虚拟质量
            desired_damping: 期望的虚拟阻尼
            desired_stiffness: 期望的虚拟刚度
            dt: 控制时间步长
        """
        self.M_d = desired_mass
        self.B_d = desired_damping
        self.K_d = desired_stiffness
        self.dt = dt
        
        # 状态变量
        self.x_d = np.zeros(3)      # 期望位置 (末端执行器)
        self.x_current = np.zeros(3) # 当前位置
        self.x_prev = np.zeros(3)   # 上一时刻位置
        self.dx_current = np.zeros(3) # 当前速度
        self.dx_prev = np.zeros(3)  # 上一时刻速度
        
        # 力传感器数据
        self.f_ext = np.zeros(3)    # 外部力
        
        # 控制增益
        self.kp = 1000.0  # 位置增益
        self.kv = 100.0   # 速度增益
        
        # 初始化标志
        self.initialized = False
        
    def set_desired_position(self, x_desired):
        """设置期望位置"""
        self.x_d = np.array(x_desired)
        
    def set_admittance_parameters(self, mass=None, damping=None, stiffness=None):
        """更新导纳参数"""
        if mass is not None:
            self.M_d = mass
        if damping is not None:
            self.B_d = damping
        if stiffness is not None:
            self.K_d = stiffness
            
    def get_end_effector_pose(self, model, data):
        """获取末端执行器位置和姿态"""
        # 获取末端执行器位置 (attachment_site)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if site_id >= 0:
            pos = data.site_xpos[site_id].copy()
            mat = data.site_xmat[site_id].reshape(3, 3).copy()
            return pos, mat
        else:
            # 如果没有找到site，使用最后一个body的位置
            return data.xpos[-1].copy(), np.eye(3)
            
    def get_external_force(self, model, data):
        """
        获取外部力
        这里简化处理，实际应用中需要从力传感器读取
        或通过雅可比矩阵从关节力矩推算
        """
        # 简化：从接触力中估算外部力
        f_ext = np.zeros(3)
        
        # 如果有接触，计算接触力
        if data.ncon > 0:
            for i in range(data.ncon):
                contact = data.contact[i]
                # 获取接触力
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                # 累加接触力的线性分量
                f_ext += force[:3]
                
        return f_ext
        
    def update_state(self, model, data):
        """更新控制器状态"""
        # 获取当前末端执行器位置
        pos, _ = self.get_end_effector_pose(model, data)
        
        if not self.initialized:
            # 初始化
            self.x_current = pos.copy()
            self.x_prev = pos.copy()
            self.x_d = pos.copy()  # 初始期望位置设为当前位置
            self.dx_current = np.zeros(3)
            self.dx_prev = np.zeros(3)
            self.initialized = True
        else:
            # 更新位置和速度
            self.x_prev = self.x_current.copy()
            self.x_current = pos.copy()
            
            self.dx_prev = self.dx_current.copy()
            self.dx_current = (self.x_current - self.x_prev) / self.dt
            
        # 获取外部力
        self.f_ext = self.get_external_force(model, data)
        
    def compute_admittance(self):
        """
        计算导纳控制
        求解微分方程: M_d * ddx + B_d * dx + K_d * (x - x_d) = F_ext
        """
        if not self.initialized:
            return np.zeros(3)
            
        # 位置误差
        pos_error = self.x_current - self.x_d
        
        # 计算期望加速度
        # ddx = (F_ext - B_d * dx - K_d * pos_error) / M_d
        ddx_desired = (self.f_ext - self.B_d * self.dx_current - self.K_d * pos_error) / self.M_d
        
        # 积分得到期望速度和位置修正
        dx_desired = self.dx_prev + ddx_desired * self.dt
        x_correction = dx_desired * self.dt
        
        return x_correction
        
    def compute_control(self, model, data, target_pos=None):
        """
        计算控制力矩
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            target_pos: 目标位置 (可选)
            
        Returns:
            控制力矩向量
        """
        # 更新状态
        self.update_state(model, data)
        
        # 如果指定了目标位置，更新期望位置
        if target_pos is not None:
            self.set_desired_position(target_pos)
            
        # 计算导纳修正
        x_correction = self.compute_admittance()
        
        # 修正期望位置
        x_modified = self.x_d + x_correction
        
        # 计算末端执行器到关节空间的雅可比矩阵
        jacp = np.zeros((3, model.nv))  # 位置雅可比
        jacr = np.zeros((3, model.nv))  # 旋转雅可比
        
        # 获取末端执行器site的ID
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if site_id >= 0:
            mujoco.mj_jac(model, data, jacp, jacr, data.site_xpos[site_id], site_id)
        else:
            # 如果没有site，使用最后一个body
            body_id = model.nbody - 1
            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
            
        # 只使用前7个关节（FR3机械臂）
        J = jacp[:, :7]
        
        # 计算期望的末端执行器速度
        pos_error = x_modified - self.x_current
        vel_desired = self.kp * pos_error / 1000.0  # 降低增益避免震荡
        
        # 通过雅可比矩阵转换为关节速度
        try:
            # 使用伪逆求解
            J_pinv = np.linalg.pinv(J)
            qvel_desired = J_pinv @ vel_desired
        except np.linalg.LinAlgError:
            qvel_desired = np.zeros(7)
            
        # PD控制计算关节力矩
        qvel_error = qvel_desired - data.qvel[:7]
        tau = self.kv * qvel_error
        
        # 限制力矩幅值
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])  # FR3关节力矩限制
        tau = np.clip(tau, -tau_max, tau_max)
        
        return tau
        
    def reset(self):
        """重置控制器状态"""
        self.x_current = np.zeros(3)
        self.x_prev = np.zeros(3)
        self.dx_current = np.zeros(3)
        self.dx_prev = np.zeros(3)
        self.f_ext = np.zeros(3)
        self.initialized = False
        
    def get_status(self):
        """获取控制器状态信息"""
        return {
            'current_position': self.x_current.copy(),
            'desired_position': self.x_d.copy(),
            'current_velocity': self.dx_current.copy(),
            'external_force': self.f_ext.copy(),
            'position_error': np.linalg.norm(self.x_current - self.x_d),
            'admittance_params': {
                'mass': self.M_d,
                'damping': self.B_d,
                'stiffness': self.K_d
            }
        }

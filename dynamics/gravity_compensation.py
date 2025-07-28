"""
重力补偿模块
实现FR3机械臂的重力补偿算法
"""

import numpy as np
import mujoco


# 重力补偿配置
GRAVITY_COMPENSATION_CONFIG = {
    'default': {
        'compensation_factor': 1.0,
        'method': 'bias'  # 'bias' 或 'inverse'
    },
    
    'partial': {
        'compensation_factor': 0.8,
        'method': 'bias'
    },
    
    'adaptive': {
        'base_factor': 1.0,
        'velocity_threshold': 0.1,
        'method': 'adaptive'
    }
}

# FR3动力学参数
FR3_DYNAMICS = {
    'joint_limits': {
        'max_torque': [87, 87, 87, 87, 12, 12, 12],  # 各关节最大力矩 (Nm)
        'max_velocity': [2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14]  # 各关节最大速度 (rad/s)
    },
    
    'inertial_properties': {
        # 这些参数从URDF/XML文件中提取
        'base_mass': 4.0,  # 估计值，实际值在XML中定义
        'total_mass': 18.0  # 估计总质量
    },
    
    'control_gains': {
        'gravity_compensation': {
            'kp': 0.0,  # 重力补偿不需要比例增益
            'kd': 0.0   # 重力补偿不需要微分增益
        }
    }
}


class GravityCompensator:
    """
    重力补偿器
    
    计算并补偿机械臂在当前配置下的重力力矩，
    使机械臂在静止状态下不会因重力而下垂。
    """
    
    def __init__(self, model, compensation_factor=1.0):
        """
        初始化重力补偿器
        
        Args:
            model: MuJoCo模型
            compensation_factor: 补偿系数 (0.0-1.0)，1.0为完全补偿
        """
        self.model = model
        self.compensation_factor = compensation_factor
        self.n_joints = 7  # FR3有7个关节
        
        # 预分配数组以提高性能
        self.gravity_torque = np.zeros(self.n_joints)
        self.temp_qfrc_bias = np.zeros(model.nv)
        
    def compute_gravity_compensation(self, data):
        """
        计算重力补偿力矩
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            重力补偿力矩向量 (7维)
        """
        # 保存当前的速度和加速度
        qvel_orig = data.qvel.copy()
        qacc_orig = data.qacc.copy()
        
        # 将速度和加速度设为零，只计算重力项
        data.qvel[:] = 0
        data.qacc[:] = 0
        
        # 计算逆动力学，得到重力和科里奥利力
        mujoco.mj_inverse(self.model, data)
        
        # 提取前7个关节的重力力矩
        self.gravity_torque = data.qfrc_inverse[:self.n_joints].copy()
        
        # 恢复原始的速度和加速度
        data.qvel[:] = qvel_orig
        data.qacc[:] = qacc_orig
        
        # 应用补偿系数
        return self.compensation_factor * self.gravity_torque
    
    def compute_gravity_compensation_bias(self, data):
        """
        使用bias力计算重力补偿（更高效的方法）
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            重力补偿力矩向量 (7维)
        """
        # 计算bias力（包含重力、科里奥利力等）
        mujoco.mj_rne(self.model, data, 1, self.temp_qfrc_bias)
        
        # 提取前7个关节的重力补偿力矩
        self.gravity_torque = -self.temp_qfrc_bias[:self.n_joints]
        
        # 应用补偿系数
        return self.compensation_factor * self.gravity_torque
    
    def set_compensation_factor(self, factor):
        """
        设置补偿系数
        
        Args:
            factor: 补偿系数 (0.0-1.0)
        """
        self.compensation_factor = np.clip(factor, 0.0, 1.0)
    
    def get_compensation_factor(self):
        """获取当前补偿系数"""
        return self.compensation_factor
    
    def enable_full_compensation(self):
        """启用完全重力补偿"""
        self.compensation_factor = 1.0
    
    def disable_compensation(self):
        """禁用重力补偿"""
        self.compensation_factor = 0.0
    
    def get_joint_gravity_torques(self, data):
        """
        获取各关节的重力力矩详细信息
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            字典，包含各关节的重力力矩信息
        """
        gravity_torques = self.compute_gravity_compensation_bias(data)
        
        joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        result = {}
        for i, name in enumerate(joint_names):
            result[name] = {
                'torque': gravity_torques[i],
                'position': data.qpos[i],
                'velocity': data.qvel[i]
            }
        
        return result


class AdaptiveGravityCompensator(GravityCompensator):
    """
    自适应重力补偿器
    
    根据机械臂的运动状态自动调整补偿系数
    """
    
    def __init__(self, model, base_factor=1.0, velocity_threshold=0.1):
        """
        初始化自适应重力补偿器
        
        Args:
            model: MuJoCo模型
            base_factor: 基础补偿系数
            velocity_threshold: 速度阈值，超过此值时减少补偿
        """
        super().__init__(model, base_factor)
        self.base_factor = base_factor
        self.velocity_threshold = velocity_threshold
        self.adaptive_factor = base_factor
        
    def compute_adaptive_factor(self, data):
        """
        根据关节速度计算自适应补偿系数
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            自适应补偿系数
        """
        # 计算关节速度的范数
        joint_velocities = data.qvel[:self.n_joints]
        velocity_norm = np.linalg.norm(joint_velocities)
        
        if velocity_norm < self.velocity_threshold:
            # 低速时使用完全补偿
            self.adaptive_factor = self.base_factor
        else:
            # 高速时减少补偿，避免干扰运动
            reduction_factor = max(0.1, 1.0 - (velocity_norm - self.velocity_threshold) / self.velocity_threshold)
            self.adaptive_factor = self.base_factor * reduction_factor
        
        return self.adaptive_factor
    
    def compute_gravity_compensation(self, data):
        """
        计算自适应重力补偿力矩
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            自适应重力补偿力矩向量
        """
        # 计算自适应系数
        adaptive_factor = self.compute_adaptive_factor(data)
        
        # 临时设置补偿系数
        original_factor = self.compensation_factor
        self.compensation_factor = adaptive_factor
        
        # 计算补偿力矩
        compensation_torques = super().compute_gravity_compensation_bias(data)
        
        # 恢复原始补偿系数
        self.compensation_factor = original_factor
        
        return compensation_torques
    
    def get_status(self, data):
        """
        获取自适应补偿器状态
        
        Args:
            data: MuJoCo数据结构
            
        Returns:
            状态字典
        """
        joint_velocities = data.qvel[:self.n_joints]
        velocity_norm = np.linalg.norm(joint_velocities)
        
        return {
            'base_factor': self.base_factor,
            'adaptive_factor': self.adaptive_factor,
            'velocity_norm': velocity_norm,
            'velocity_threshold': self.velocity_threshold,
            'joint_velocities': joint_velocities.copy()
        }

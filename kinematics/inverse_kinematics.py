"""
逆运动学模块
使用数值方法计算达到目标位姿所需的关节角度
"""

import numpy as np
import mujoco
from .forward_kinematics import ForwardKinematics


class InverseKinematics:
    """逆运动学类，用于计算达到目标末端执行器位姿所需的关节角度"""
    
    def __init__(self, model, data):
        """
        初始化逆运动学计算器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
        """
        self.model = model
        self.data = data
        self.fk = ForwardKinematics(model, data)
        
        # 关节限制
        self.joint_limits_lower = model.jnt_range[:7, 0]  # 下限
        self.joint_limits_upper = model.jnt_range[:7, 1]  # 上限
    
    def solve_ik(self, target_position, target_orientation=None, initial_guess=None, 
                 max_iterations=100, tolerance=1e-4, damping=0.1):
        """
        求解逆运动学
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标旋转矩阵 (3x3) 或四元数 [w, x, y, z]，可选
            initial_guess: 初始关节角度猜测，如果为None则使用当前角度
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            damping: 阻尼系数，用于数值稳定性
            
        Returns:
            tuple: (joint_angles, success, error)
                - joint_angles: 计算得到的关节角度
                - success: 是否成功收敛
                - error: 最终误差
        """
        # 初始化关节角度
        if initial_guess is None:
            q = self.data.qpos[:7].copy()
        else:
            q = np.array(initial_guess)
        
        # 目标位姿
        target_pos = np.array(target_position)
        
        # 处理目标姿态
        if target_orientation is not None:
            if len(target_orientation) == 4:  # 四元数
                target_rot_mat = self._quaternion_to_rotation_matrix(target_orientation)
            else:  # 旋转矩阵
                target_rot_mat = np.array(target_orientation)
            use_orientation = True
        else:
            use_orientation = False
        
        for iteration in range(max_iterations):
            # 计算当前位姿
            current_pos, current_rot_mat = self.fk.compute_fk(q)
            
            # 计算位置误差
            pos_error = target_pos - current_pos
            
            # 计算姿态误差（如果指定了目标姿态）
            if use_orientation:
                # 使用轴角表示计算姿态误差
                rot_error = self._rotation_error(current_rot_mat, target_rot_mat)
                error_vector = np.concatenate([pos_error, rot_error])
            else:
                error_vector = pos_error
            
            # 检查收敛
            error_norm = np.linalg.norm(error_vector)
            if error_norm < tolerance:
                return q, True, error_norm
            
            # 计算雅可比矩阵
            jacobian = self.fk.get_jacobian(q)
            
            if not use_orientation:
                # 只使用位置部分的雅可比
                jacobian = jacobian[:3, :]
            
            # 使用阻尼最小二乘法求解，改进奇异性处理
            # Δq = J^T(JJ^T + λI)^(-1) * error
            JJT = jacobian @ jacobian.T
            
            # 自适应阻尼：如果雅可比接近奇异，增加阻尼
            condition_number = np.linalg.cond(jacobian)
            if condition_number > 1e6:  # 如果条件数太大
                adaptive_damping = damping * condition_number / 1e6
            else:
                adaptive_damping = damping
                
            damping_matrix = adaptive_damping * np.eye(JJT.shape[0])
            
            try:
                delta_q = jacobian.T @ np.linalg.solve(JJT + damping_matrix, error_vector)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                pinv_jacobian = np.linalg.pinv(jacobian)
                delta_q = pinv_jacobian @ error_vector
            
            # 更新关节角度
            q += delta_q
            
            # 应用关节限制
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
        
        # 未收敛
        final_error = np.linalg.norm(error_vector)
        return q, False, final_error
    
    def solve_ik_position_only(self, target_position, initial_guess=None, 
                              max_iterations=100, tolerance=1e-4, damping=0.1):
        """
        仅针对位置的逆运动学求解
        
        Args:
            target_position: 目标位置 [x, y, z]
            initial_guess: 初始关节角度猜测
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            damping: 阻尼系数
            
        Returns:
            tuple: (joint_angles, success, error)
        """
        return self.solve_ik(target_position, None, initial_guess, 
                           max_iterations, tolerance, damping)
    
    def solve_ik_multiple_targets(self, target_positions, target_orientations=None, 
                                 initial_guess=None):
        """
        求解多个目标点的逆运动学
        
        Args:
            target_positions: 目标位置列表
            target_orientations: 目标姿态列表，可选
            initial_guess: 初始关节角度猜测
            
        Returns:
            list: 每个目标的求解结果 [(joint_angles, success, error), ...]
        """
        results = []
        current_q = initial_guess if initial_guess is not None else self.data.qpos[:7].copy()
        
        for i, target_pos in enumerate(target_positions):
            target_ori = target_orientations[i] if target_orientations else None
            
            q, success, error = self.solve_ik(target_pos, target_ori, current_q)
            results.append((q, success, error))
            
            # 使用上一个解作为下一个的初始猜测
            current_q = q
        
        return results
    
    def _quaternion_to_rotation_matrix(self, q):
        """
        将四元数转换为旋转矩阵
        
        Args:
            q: 四元数 [w, x, y, z]
            
        Returns:
            numpy.ndarray: 3x3旋转矩阵
        """
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def _rotation_error(self, R_current, R_target):
        """
        计算旋转误差（轴角表示）
        
        Args:
            R_current: 当前旋转矩阵
            R_target: 目标旋转矩阵
            
        Returns:
            numpy.ndarray: 3D旋转误差向量
        """
        # 计算相对旋转
        R_error = R_target @ R_current.T
        
        # 转换为轴角表示
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        
        if np.abs(angle) < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(angle))
        
        return angle * axis
    
    def get_reachable_workspace(self, num_samples=1000):
        """
        计算机器人的可达工作空间
        
        Args:
            num_samples: 采样点数量
            
        Returns:
            numpy.ndarray: 可达位置点集合 (num_samples, 3)
        """
        positions = []
        
        for _ in range(num_samples):
            # 随机生成关节角度
            random_angles = np.random.uniform(
                self.joint_limits_lower, 
                self.joint_limits_upper
            )
            
            # 计算对应的末端执行器位置
            pos, _ = self.fk.compute_fk(random_angles)
            positions.append(pos)
        
        return np.array(positions)

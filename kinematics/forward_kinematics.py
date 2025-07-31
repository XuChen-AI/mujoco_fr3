"""
正运动学模块
使用MuJoCo计算机器人末端执行器的位置和姿态
"""

import numpy as np
import mujoco


class ForwardKinematics:
    """正运动学类，用于计算给定关节角度下的末端执行器位姿"""
    
    def __init__(self, model, data):
        """
        初始化正运动学计算器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
        """
        self.model = model
        self.data = data
        
        # 找到末端执行器的body ID
        try:
            self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_link7')
        except:
            # 如果找不到fr3_link7，尝试其他可能的名称
            try:
                self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'panda_hand')
            except:
                # 如果都找不到，使用最后一个body
                self.end_effector_id = model.nbody - 1
                print(f"警告: 无法找到末端执行器，使用body ID: {self.end_effector_id}")
                
        # 查找attachment_site用于更精确的末端执行器位置
        try:
            self.attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
            self.use_site = True
        except:
            self.use_site = False
    
    def compute_fk(self, joint_angles):
        """
        计算正运动学
        
        Args:
            joint_angles: 关节角度数组 (7个元素)
            
        Returns:
            tuple: (position, orientation_matrix)
                - position: 3D位置向量 [x, y, z]
                - orientation_matrix: 3x3旋转矩阵
        """
        # 保存当前关节角度
        original_qpos = self.data.qpos[:7].copy()
        
        # 设置关节角度
        self.data.qpos[:7] = joint_angles
        
        # 执行前向运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 获取末端执行器位置
        if self.use_site:
            # 使用site位置，通常更精确
            position = self.data.site_xpos[self.attachment_site_id].copy()
        else:
            # 使用body位置
            position = self.data.xpos[self.end_effector_id].copy()
        
        # 获取末端执行器旋转矩阵
        rotation_matrix = self.data.xmat[self.end_effector_id].reshape(3, 3).copy()
        
        # 恢复原始关节角度
        self.data.qpos[:7] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return position, rotation_matrix
    
    def compute_fk_with_quaternion(self, joint_angles):
        """
        计算正运动学（四元数表示）
        
        Args:
            joint_angles: 关节角度数组 (7个元素)
            
        Returns:
            tuple: (position, quaternion)
                - position: 3D位置向量 [x, y, z]
                - quaternion: 四元数 [w, x, y, z]
        """
        position, rotation_matrix = self.compute_fk(joint_angles)
        
        # 将旋转矩阵转换为四元数
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        return position, quaternion
    
    def get_jacobian(self, joint_angles):
        """
        计算雅可比矩阵
        
        Args:
            joint_angles: 关节角度数组 (7个元素)
            
        Returns:
            numpy.ndarray: 6x7雅可比矩阵 (前3行为位置，后3行为姿态)
        """
        # 保存当前关节角度
        original_qpos = self.data.qpos[:7].copy()
        
        # 设置关节角度
        self.data.qpos[:7] = joint_angles
        mujoco.mj_forward(self.model, self.data)
        
        # 初始化雅可比矩阵
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        
        # 计算雅可比矩阵
        if self.use_site:
            # 对site计算雅可比
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.attachment_site_id)
        else:
            # 对body计算雅可比
            mujoco.mj_jac(self.model, self.data, jac_pos, jac_rot, 
                         self.data.xpos[self.end_effector_id], self.end_effector_id)
        
        # 恢复原始关节角度
        self.data.qpos[:7] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # 只返回前7列（对应7个关节）
        jacobian = np.vstack([jac_pos[:, :7], jac_rot[:, :7]])
        
        return jacobian
    
    def _rotation_matrix_to_quaternion(self, R):
        """
        将旋转矩阵转换为四元数
        
        Args:
            R: 3x3旋转矩阵
            
        Returns:
            numpy.ndarray: 四元数 [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])

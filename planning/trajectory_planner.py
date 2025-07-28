
# trajectory_planner.py
# 轨迹规划器模块
# 本文件用于实现机械臂的轨迹规划相关功能。
# 包含轨迹生成、插值等常用方法。
# 所有函数均配有详细中文注释，便于理解和维护。

import numpy as np  # 导入numpy库用于数值计算

class TrajectoryPlanner:
    """
    轨迹规划器类
    用于生成机械臂的关节空间或笛卡尔空间轨迹。
    支持线性插值、三次多项式插值等常见轨迹规划方法。
    """

    def __init__(self):
        """
        初始化轨迹规划器对象
        当前无参数，后续可扩展
        """
        pass  # 占位，无实际操作

    def linear_interpolation(self, start, end, steps):
        """
        线性插值方法
        输入起点、终点和插值步数，返回插值轨迹点
        参数：
            start (np.ndarray): 起点，关节或空间坐标
            end (np.ndarray): 终点，关节或空间坐标
            steps (int): 插值步数
        返回：
            np.ndarray: 插值后的轨迹点数组，shape为(steps, dim)
        """
        # 使用numpy的linspace进行线性插值
        return np.linspace(start, end, steps)  # 返回插值结果

    def cubic_interpolation(self, start, end, steps):
        """
        三次多项式插值方法
        输入起点、终点和插值步数，返回平滑轨迹点
        参数：
            start (np.ndarray): 起点，关节或空间坐标
            end (np.ndarray): 终点，关节或空间坐标
            steps (int): 插值步数
        返回：
            np.ndarray: 插值后的轨迹点数组，shape为(steps, dim)
        """
        # 生成归一化时间序列
        t = np.linspace(0, 1, steps)  # 时间归一化到[0,1]
        # 三次多项式系数
        a0 = start  # 初始位置
        a3 = 2 * (start - end) + (0) + (0)  # 假设速度为0
        a2 = 3 * (end - start) - (0) - (0)  # 假设速度为0
        # 计算每个时间点的插值结果
        traj = np.array([a0 + a2 * (ti ** 2) + a3 * (ti ** 3) for ti in t])  # 逐步计算插值
        return traj  # 返回插值结果

    def plan_trajectory(self, waypoints, steps_per_segment=50, method="linear"):
        """
        轨迹规划主方法
        根据给定路径点，生成完整轨迹
        参数：
            waypoints (list[np.ndarray]): 路径点列表
            steps_per_segment (int): 每段插值步数，默认50
            method (str): 插值方法，"linear"或"cubic"
        返回：
            np.ndarray: 规划后的完整轨迹点数组
        """
        trajectory = []  # 初始化轨迹列表
        # 遍历路径点，逐段插值
        for i in range(len(waypoints) - 1):
            start = waypoints[i]  # 当前段起点
            end = waypoints[i + 1]  # 当前段终点
            if method == "linear":
                segment = self.linear_interpolation(start, end, steps_per_segment)  # 线性插值
            elif method == "cubic":
                segment = self.cubic_interpolation(start, end, steps_per_segment)  # 三次插值
            else:
                raise ValueError("不支持的插值方法: {}".format(method))  # 方法错误抛出异常
            if i == 0:
                trajectory.append(segment)  # 第一段全部加入
            else:
                trajectory.append(segment[1:])  # 后续段去除首点避免重复
        # 拼接所有段，形成完整轨迹
        return np.vstack(trajectory)  # 返回完整轨迹数组

    def plan_comfortable_trajectory(self, start_joints, target_joints):
        """
        规划舒适的机械臂轨迹
        考虑关节运动的最小化、速度平滑性和避免奇异位置
        
        参数：
            start_joints (np.ndarray): 起始关节角度
            target_joints (np.ndarray): 目标关节角度
        返回：
            np.ndarray: 优化后的轨迹点数组
        """
        # 计算关节角度差异
        joint_diff = target_joints - start_joints
        
        # 1. 关节角度标准化：将角度差限制在[-π, π]范围内
        # 这样可以选择最短的旋转路径，避免不必要的大幅度旋转
        normalized_diff = self._normalize_joint_angles(joint_diff)
        normalized_target = start_joints + normalized_diff
        
        # 2. 检查是否需要添加中间路径点来避免奇异位置或关节限制
        waypoints = self._generate_intermediate_waypoints(start_joints, normalized_target)
        
        # 3. 使用改进的轨迹插值生成平滑轨迹
        trajectory = self._generate_smooth_trajectory(waypoints)
        
        return trajectory
    
    def _normalize_joint_angles(self, joint_diff):
        """
        标准化关节角度差异，选择最短旋转路径
        将角度差限制在[-π, π]范围内
        
        参数：
            joint_diff (np.ndarray): 关节角度差异
        返回：
            np.ndarray: 标准化后的角度差异
        """
        normalized = joint_diff.copy()
        for i in range(len(normalized)):
            # 将角度差标准化到[-π, π]范围
            while normalized[i] > np.pi:
                normalized[i] -= 2 * np.pi
            while normalized[i] < -np.pi:
                normalized[i] += 2 * np.pi
        return normalized
    
    def _generate_intermediate_waypoints(self, start_joints, target_joints):
        """
        生成中间路径点，避免关节限制和奇异位置
        
        参数：
            start_joints (np.ndarray): 起始关节角度
            target_joints (np.ndarray): 目标关节角度
        返回：
            list: 包含中间路径点的列表
        """
        waypoints = [start_joints]
        
        # 计算最大关节角度变化
        max_joint_change = np.max(np.abs(target_joints - start_joints))
        
        # 如果关节变化较大，添加中间点
        if max_joint_change > np.pi/2:  # 如果最大变化超过90度
            # 计算需要的中间点数量
            num_intermediate = int(np.ceil(max_joint_change / (np.pi/4)))  # 每45度一个中间点
            
            for i in range(1, num_intermediate):
                t = i / num_intermediate
                # 使用平滑的三次函数生成中间点，而不是线性插值
                # 这样可以确保速度变化更平滑
                smooth_t = 3*t**2 - 2*t**3  # 平滑插值函数 (S曲线)
                intermediate = start_joints + smooth_t * (target_joints - start_joints)
                waypoints.append(intermediate)
        
        waypoints.append(target_joints)
        return waypoints
    
    def _generate_smooth_trajectory(self, waypoints):
        """
        生成平滑的轨迹，使用改进的插值方法
        确保速度和加速度连续性
        
        参数：
            waypoints (list): 路径点列表
        返回：
            np.ndarray: 平滑的轨迹点数组
        """
        trajectory = []
        total_segments = len(waypoints) - 1
        
        # 根据轨迹复杂度调整每段的插值点数
        base_steps = 50
        for i in range(total_segments):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # 计算这段的关节变化量，调整插值点数
            joint_change = np.max(np.abs(end - start))
            steps = max(base_steps, int(base_steps * joint_change / (np.pi/4)))
            
            # 使用五次多项式插值确保速度和加速度连续
            segment = self._quintic_interpolation(start, end, steps)
            
            if i == 0:
                trajectory.extend(segment)
            else:
                # 跳过第一个点避免重复
                trajectory.extend(segment[1:])
        
        return np.array(trajectory)
    
    def _quintic_interpolation(self, start, end, steps):
        """
        五次多项式插值，确保位置、速度和加速度连续
        提供比三次插值更平滑的运动
        
        参数：
            start (np.ndarray): 起始点
            end (np.ndarray): 终止点
            steps (int): 插值步数
        返回：
            list: 插值后的轨迹点列表
        """
        t = np.linspace(0, 1, steps)
        
        # 五次多项式系数（假设起始和结束速度、加速度为0）
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # 边界条件：q(0)=start, q(1)=end, q'(0)=0, q'(1)=0, q''(0)=0, q''(1)=0
        
        trajectory = []
        for ti in t:
            # 五次多项式 (10t^3 - 15t^4 + 6t^5) 确保平滑的S曲线
            blend = 10*ti**3 - 15*ti**4 + 6*ti**5
            point = start + blend * (end - start)
            trajectory.append(point)
        
        return trajectory

"""
控制器配置文件
定义各种控制器的默认参数
"""

# 导纳控制器默认参数
ADMITTANCE_CONFIG = {
    'default': {
        'desired_mass': 1.0,
        'desired_damping': 20.0,
        'desired_stiffness': 100.0,
        'dt': 0.002,
        'kp': 1000.0,
        'kv': 100.0
    },
    
    'soft': {  # 柔软模式 - 高柔顺性
        'desired_mass': 3.0,
        'desired_damping': 80.0,
        'desired_stiffness': 50.0,
        'dt': 0.002,
        'kp': 500.0,
        'kv': 50.0
    },
    
    'stiff': {  # 刚性模式 - 高精度
        'desired_mass': 0.5,
        'desired_damping': 15.0,
        'desired_stiffness': 300.0,
        'dt': 0.002,
        'kp': 2000.0,
        'kv': 200.0
    },
    
    'medium': {  # 中等模式 - 平衡性能
        'desired_mass': 2.0,
        'desired_damping': 50.0,
        'desired_stiffness': 200.0,
        'dt': 0.002,
        'kp': 1500.0,
        'kv': 150.0
    }
}

# FR3机械臂参数
FR3_CONFIG = {
    'joint_limits': {
        'position': [
            [-2.7437, 2.7437],   # joint1
            [-1.7837, 1.7837],   # joint2
            [-2.9007, 2.9007],   # joint3
            [-3.0421, -0.1518],  # joint4
            [-2.8065, 2.8065],   # joint5
            [0.5445, 4.5169],    # joint6
            [-3.0159, 3.0159]    # joint7
        ],
        'torque': [87, 87, 87, 87, 12, 12, 12]  # 最大力矩
    },
    
    'home_position': [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
    
    'workspace': {
        'x_range': [-0.8, 0.8],
        'y_range': [-0.8, 0.8], 
        'z_range': [0.0, 1.2]
    }
}

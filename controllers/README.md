# 控制器模块

这个文件夹包含了FR3机械臂的各种控制算法实现。

## 文件结构

```
controllers/
├── __init__.py                 # 模块初始化
├── admittance_controller.py    # 导纳控制器实现
├── config.py                   # 控制器配置参数
└── README.md                   # 说明文档
```

## 导纳控制器 (Admittance Controller)

导纳控制是一种力控制方法，使机械臂在接触时表现出柔顺性。控制器通过调节机械臂的位置和速度来响应外部力，使机械臂表现出指定的动态特性。

### 控制原理

导纳控制基于以下微分方程：
```
M_d * ẍ + B_d * ẋ + K_d * (x - x_d) = F_ext
```

其中：
- `M_d`: 期望惯性矩阵 (虚拟质量)
- `B_d`: 期望阻尼矩阵 (虚拟阻尼)
- `K_d`: 期望刚度矩阵 (虚拟刚度)
- `F_ext`: 外部力
- `x_d`: 期望位置

### 使用方法

#### 基本使用

```python
from controllers.admittance_controller import AdmittanceController

# 创建控制器实例
admittance_ctrl = AdmittanceController(
    desired_mass=2.0,       # 虚拟质量
    desired_damping=50.0,   # 虚拟阻尼
    desired_stiffness=200.0, # 虚拟刚度
    dt=0.002               # 控制周期
)

# 在仿真循环中使用
while running:
    # 计算控制力矩
    tau = admittance_ctrl.compute_control(model, data, target_position)
    
    # 应用到机械臂
    data.ctrl[:7] = tau
    
    # 执行仿真步骤
    mujoco.mj_step(model, data)
```

#### 使用预设配置

```python
from controllers.admittance_controller import AdmittanceController
from controllers.config import ADMITTANCE_CONFIG

# 使用预设的"软"模式配置
config = ADMITTANCE_CONFIG['soft']
admittance_ctrl = AdmittanceController(**config)
```

### 预设配置模式

- **`default`**: 默认平衡设置
- **`soft`**: 高柔顺性，适合安全交互
- **`stiff`**: 高精度，适合精确定位
- **`medium`**: 平衡性能

### 参数调节指南

- **虚拟质量 (desired_mass)**:
  - 增大：机械臂响应更平滑，但反应较慢
  - 减小：机械臂响应更快，但可能震荡

- **虚拟阻尼 (desired_damping)**:
  - 增大：减少震荡，但响应变慢
  - 减小：响应更快，但可能不稳定

- **虚拟刚度 (desired_stiffness)**:
  - 增大：位置保持能力强，但柔顺性差
  - 减小：柔顺性好，但位置精度降低

## 运行示例

使用以下命令运行导纳控制演示：

```bash
python admittance_demo.py
```

这个演示程序会：
1. 启动FR3机械臂仿真
2. 应用导纳控制器
3. 让末端执行器执行缓慢的轨迹跟踪
4. 响应外部接触力（可以用鼠标在仿真中推动机械臂）

## 扩展功能

### 添加新的控制器

1. 在`controllers`文件夹中创建新的Python文件
2. 实现控制器类
3. 在`__init__.py`中导入新控制器
4. 在`config.py`中添加相关配置

### 自定义参数

可以通过修改`config.py`文件来调整控制器参数，或者在运行时动态设置：

```python
# 运行时调整参数
admittance_ctrl.set_admittance_parameters(
    mass=3.0,
    damping=60.0,
    stiffness=150.0
)
```

## 注意事项

1. **安全性**: 在实际机器人上使用前，请仔细调试参数
2. **稳定性**: 过低的阻尼可能导致系统不稳定
3. **性能**: 控制频率应匹配仿真步长
4. **力反馈**: 当前实现使用简化的力估算，实际应用需要真实的力传感器数据

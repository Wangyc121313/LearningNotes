# ROS 2 — Robot Operating System 2

> 机器人操作系统 ROS 2 学习笔记

## 内容概览

- ROS 2 核心概念（Node、Topic、Service、Action）
- 通信机制（DDS、QoS）
- 常用工具（colcon、rviz2、rqt）
- 与 VLA 模型的集成：将动作指令部署到真实机器人
- Nav2 导航与 MoveIt 2 运动规划

## 1. 核心概念

ROS 2 是一个机器人软件框架，提供节点间通信、硬件抽象、工具链等基础设施。

### 核心通信原语

| 原语 | 方向 | 同步性 | 适用场景 |
|------|------|--------|---------|
| **Topic（话题）** | 发布/订阅（1对多） | 异步 | 传感器数据流（图像、激光雷达、IMU）|
| **Service（服务）** | 请求/响应（1对1） | 同步 | 一次性任务（坐标变换查询、参数获取）|
| **Action（动作）** | 目标/反馈/结果 | 异步+反馈 | 长时间任务（导航、机械臂运动）|
| **Parameter（参数）** | 节点内部配置 | 动态可修改 | 控制器增益、话题名称等 |

### ROS 2 vs ROS 1 主要区别

| 特性 | ROS 1 | ROS 2 |
|------|-------|-------|
| 通信中间件 | 自研 TCPROS | **DDS**（Data Distribution Service） |
| Master 节点 | 需要 roscore | **无需 Master** |
| 实时性 | 弱 | 支持实时（RTOS / real-time Linux） |
| 安全 | 无 | DDS 安全（加密/认证）|
| 多语言 | Python/C++ | Python/C++/Rust/... |

---

## 2. DDS 与 QoS

**DDS（Data Distribution Service）**：工业级发布/订阅中间件，是 ROS 2 的通信基础。

### QoS（Quality of Service）关键策略

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# 传感器数据（允许丢包，低延迟优先）
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,    # 尽力传输，不保证到达
    durability=DurabilityPolicy.VOLATILE,          # 不缓存历史消息
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# 状态/命令（必须可靠）
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,        # 保证到达（类似 TCP）
    durability=DurabilityPolicy.TRANSIENT_LOCAL,   # 新订阅者可收到最后一条消息
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
```

**常见 QoS 预设**：
- `qos_profile_sensor_data`：BEST_EFFORT + VOLATILE，适合传感器
- `qos_profile_system_default`：RELIABLE + VOLATILE，适合系统消息

---

## 3. 核心编程模式

### 发布者（Publisher）

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import numpy as np

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # 声明参数（可通过命令行或 YAML 覆盖）
        self.declare_parameter('control_rate', 50.0)   # Hz
        self.declare_parameter('max_velocity', 1.0)

        rate = self.get_parameter('control_rate').value

        # 发布器
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/joint_commands',
            10
        )

        # 定时器：以指定频率发布
        self.timer = self.create_timer(1.0 / rate, self.control_loop)
        self.step = 0
        self.get_logger().info(f'ArmController 启动，控制频率: {rate} Hz')

    def control_loop(self):
        msg = Float64MultiArray()
        # 模拟关节角度命令（6关节正弦运动）
        t = self.step * 0.02
        msg.data = [0.5 * np.sin(t + i * 0.5) for i in range(6)]
        self.cmd_pub.publish(msg)
        self.step += 1
```

### 订阅者（Subscriber）

```python
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')
        self.bridge = CvBridge()

        # 多话题订阅
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data   # 传感器 QoS
        )

    def joint_callback(self, msg: JointState):
        # msg.name: 关节名称列表
        # msg.position: 当前角度（rad）
        # msg.velocity: 当前速度（rad/s）
        pos = dict(zip(msg.name, msg.position))
        self.get_logger().debug(f'关节角度: {pos}')

    def image_callback(self, msg: Image):
        # ROS Image → OpenCV ndarray
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h, w = cv_image.shape[:2]
        self.get_logger().info(f'收到图像: {w}x{h}')
```

### 服务（Service）

```python
from std_srvs.srv import Trigger
from custom_interfaces.srv import GetPose

class PoseService(Node):
    def __init__(self):
        super().__init__('pose_service')
        # 创建服务端
        self.srv = self.create_service(Trigger, '/get_current_pose', self.handle_pose)

    def handle_pose(self, request, response):
        response.success = True
        response.message = "x=0.5, y=0.3, z=0.8"
        return response

# 客户端调用（同步）
class PoseClient(Node):
    def __init__(self):
        super().__init__('pose_client')
        self.client = self.create_client(Trigger, '/get_current_pose')
        self.client.wait_for_service(timeout_sec=5.0)

    def call(self):
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

### Action（长时间任务）

```python
from rclpy.action import ActionServer, ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class NavClient(Node):
    def __init__(self):
        super().__init__('nav_client')
        self._client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def send_goal(self, x, y, yaw=0.0):
        self._client.wait_for_server()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_cb
        )
        future.add_done_callback(self.goal_response_cb)

    def feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'剩余距离: {fb.distance_remaining:.2f} m'
        )

    def goal_response_cb(self, future):
        result_future = future.result().get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result().result
        self.get_logger().info('导航完成！')
```

---

## 4. 常用工具

### colcon（构建工具）

```bash
# 构建工作空间
colcon build                                  # 构建所有包
colcon build --packages-select my_pkg         # 只构建指定包
colcon build --symlink-install                # 符号链接安装（Python 包修改后无需重新构建）
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release  # Release 模式

# 测试
colcon test --packages-select my_pkg
colcon test-result --verbose

# 激活工作空间
source install/setup.bash   # bash
source install/setup.zsh    # zsh
```

### 常用命令行工具

```bash
# 节点
ros2 node list                     # 列出运行中节点
ros2 node info /my_node            # 查看节点详情（发布/订阅/服务/动作）

# 话题
ros2 topic list                    # 列出所有话题
ros2 topic echo /joint_states      # 实时打印话题消息
ros2 topic hz /camera/image_raw    # 统计话题频率
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"  # 手动发布

# 服务
ros2 service list
ros2 service call /get_pose std_srvs/srv/Trigger {}

# 参数
ros2 param list /my_node
ros2 param get /my_node control_rate
ros2 param set /my_node control_rate 100.0

# bag 录制与回放（调试神器）
ros2 bag record -a -o my_recording        # 录制所有话题
ros2 bag record /joint_states /camera/image_raw  # 录制指定话题
ros2 bag play my_recording                # 回放
ros2 bag info my_recording                # 查看统计信息
```

### rviz2（可视化）

```bash
rviz2                              # 打开可视化工具
# 可视化: 机器人模型(URDF) / 点云 / 坐标系 / 路径 / 图像 / 传感器数据
```

---

## 5. 坐标变换（tf2）

tf2 维护机器人各坐标系之间的变换树（Transform Tree）：

```
map → odom → base_link → base_laser
                       → camera_link → camera_optical_frame
                       → arm_base → arm_link1 → ... → end_effector
```

```python
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class TFUser(Node):
    def __init__(self):
        super().__init__('tf_user')
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_ee_in_map(self):
        """查询末端执行器在 map 系中的位置"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',               # 目标坐标系
                'end_effector',      # 源坐标系
                rclpy.time.Time(),   # 查询最新变换
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            t = transform.transform.translation
            self.get_logger().info(f'末端执行器位置: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}')
            return transform
        except Exception as e:
            self.get_logger().warn(f'TF 查询失败: {e}')
            return None
```

---

## 6. Nav2 导航框架

Nav2 是 ROS 2 官方导航栈，支持差速底盘 / 全向轮 / 阿克曼等多种机器人。

```
地图 (map) + 激光雷达/相机
        │
        ▼
  SLAM（定位建图）: slam_toolbox / cartographer
        │  全局位姿 (map→odom)
        ▼
  Nav2 Stack
  ├── Global Costmap（全局代价地图）
  ├── Global Planner（A* / NavFn 全局路径规划）
  ├── Local Costmap（局部代价地图）
  ├── Local Planner（DWB / TEB 局部避障）
  └── Recovery Behaviors（清除代价地图、原地旋转）
        │  /cmd_vel（速度命令）
        ▼
  底盘控制器
```

```bash
# 启动 Nav2（通常通过 launch 文件）
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true map:=my_map.yaml
```

---

## 7. 与 VLA 集成：完整节点模板

```python
#!/usr/bin/env python3
"""VLA + ROS 2 集成节点：图像 + 语言指令 → 机械臂动作"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import numpy as np

class VLAControlNode(Node):
    def __init__(self):
        super().__init__('vla_control_node')

        # 参数
        self.declare_parameter('model_name', 'openvla/openvla-7b')
        self.declare_parameter('inference_rate', 5.0)   # Hz（VLA 推理较慢）
        self.declare_parameter('action_scale', 0.05)    # 动作缩放系数
        self.declare_parameter('joint_names',
            ['joint1','joint2','joint3','joint4','joint5','joint6'])

        self.bridge = CvBridge()
        self.latest_image = None
        self.current_instruction = ""

        # 订阅
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._image_cb, 10)
        self.create_subscription(String, '/vla/instruction',
                                 self._instruction_cb, 10)

        # 发布
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/arm/joint_trajectory', 10)
        self.status_pub = self.create_publisher(
            String, '/vla/status', 10)

        # 推理定时器
        rate = self.get_parameter('inference_rate').value
        self.timer = self.create_timer(1.0 / rate, self._inference_loop)

        # 加载 VLA 模型
        self._load_model()
        self.get_logger().info('VLA Control Node 启动完成')

    def _load_model(self):
        model_name = self.get_parameter('model_name').value
        self.get_logger().info(f'加载模型: {model_name}')
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to('cuda')
            self.get_logger().info('模型加载完成')
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')
            self.model = None

    def _image_cb(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def _instruction_cb(self, msg: String):
        self.current_instruction = msg.data
        self.get_logger().info(f'收到指令: {msg.data}')

    def _inference_loop(self):
        if self.latest_image is None or not self.current_instruction:
            return
        if self.model is None:
            return

        try:
            import torch
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(self.latest_image)
            inputs = self.processor(
                self.current_instruction, pil_img
            ).to('cuda', dtype=torch.bfloat16)

            action = self.model.predict_action(**inputs,
                                               unnorm_key='bridge_orig',
                                               do_sample=False)
            # action: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
            self._publish_action(action)

        except Exception as e:
            self.get_logger().error(f'推理失败: {e}')

    def _publish_action(self, action: np.ndarray):
        scale = self.get_parameter('action_scale').value
        joint_names = self.get_parameter('joint_names').value

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = (action[:6] * scale).tolist()
        point.time_from_start = Duration(nanosec=200_000_000)  # 200ms
        traj.points = [point]

        self.traj_pub.publish(traj)

        status = String()
        status.data = f'动作已发布 | 夹爪: {"open" if action[6] > 0.5 else "close"}'
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = VLAControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 对应 Launch 文件

```python
# launch/vla_control.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_name', default_value='openvla/openvla-7b'),
        DeclareLaunchArgument('inference_rate', default_value='5.0'),

        Node(
            package='my_vla_pkg',
            executable='vla_control_node',
            name='vla_control',
            parameters=[{
                'model_name': LaunchConfiguration('model_name'),
                'inference_rate': LaunchConfiguration('inference_rate'),
            }],
            remappings=[
                ('/camera/color/image_raw', '/realsense/color/image_raw'),
            ]
        ),
    ])
```

```python
"""
ROS 2 代码演示
由于 ROS 2 需要专门的运行环境（Linux + ROS 2 安装），
以下代码演示核心 API 用法，并在无 ROS 2 环境时以 Mock 方式展示逻辑。
"""

import sys

# 检测 ROS 2 环境
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
    print("✅ ROS 2 环境可用")
except ImportError:
    ROS2_AVAILABLE = False
    print("ℹ️  ROS 2 未安装，以 Mock 模式演示核心逻辑\n")

# ===== Mock 类（无 ROS 2 时用于演示） =====
if not ROS2_AVAILABLE:
    import time
    import threading
    import queue

    class MockLogger:
        def info(self, msg):  print(f"[INFO ] {msg}")
        def warn(self, msg):  print(f"[WARN ] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): pass

    class MockPublisher:
        def __init__(self, topic):
            self.topic = topic
            self._count = 0
        def publish(self, msg):
            self._count += 1
            print(f"  📤 [{self.topic}] #{self._count}: {msg}")

    class MockSubscription:
        pass

    class MockNode:
        """模拟 rclpy.node.Node 的核心接口"""
        def __init__(self, name):
            self._name = name
            self._logger = MockLogger()
            self._params = {}
        def get_logger(self): return self._logger
        def declare_parameter(self, name, default): self._params[name] = default
        def get_parameter(self, name):
            class P:
                def __init__(self, v): self.value = v
            return P(self._params.get(name))
        def create_publisher(self, msg_type, topic, qos):
            return MockPublisher(topic)
        def create_subscription(self, *args, **kwargs):
            return MockSubscription()
        def destroy_node(self): pass

    Node = MockNode  # 替换 Node

# ===== 1. 发布者/订阅者基本模式 =====
print("=" * 60)
print("1. 发布者 / 订阅者基本模式")
print("=" * 60)

class MinimalPublisher(Node):
    """最简发布者：每次 run() 发布一条消息"""
    def __init__(self):
        super().__init__('minimal_publisher')
        self._pub = self.create_publisher(object, '/chatter', 10)
        self._count = 0

    def run(self, n=3):
        for _ in range(n):
            msg = f"Hello ROS 2! count={self._count}"
            self._pub.publish(msg)
            self._count += 1

class MinimalSubscriber(Node):
    """最简订阅者：收到消息后记录"""
    def __init__(self):
        super().__init__('minimal_subscriber')
        self._sub = self.create_subscription(
            object, '/chatter', self.listener_cb, 10
        )
        self.received = []

    def listener_cb(self, msg):
        self.get_logger().info(f'收到消息: {msg}')
        self.received.append(msg)

pub = MinimalPublisher()
sub = MinimalSubscriber()
pub.run(3)
print()

# ===== 2. 参数声明与读取 =====
print("=" * 60)
print("2. 节点参数")
print("=" * 60)

class ParameterDemo(Node):
    def __init__(self):
        super().__init__('param_demo')
        # 声明参数（名称, 默认值）
        self.declare_parameter('robot_name', 'arm_robot')
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('joint_limits', [(-3.14, 3.14)] * 6)

        # 读取参数
        name  = self.get_parameter('robot_name').value
        rate  = self.get_parameter('control_rate').value
        debug = self.get_parameter('debug_mode').value
        self.get_logger().info(
            f'robot={name}, rate={rate}Hz, debug={debug}'
        )

param_node = ParameterDemo()
print()

# ===== 3. 服务模式（Mock） =====
print("=" * 60)
print("3. 服务（Request / Response）")
print("=" * 60)

class MockRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class MockResponse:
    def __init__(self):
        self.success = False
        self.message = ""
    def __repr__(self):
        return f"Response(success={self.success}, message='{self.message}')"

class GripperService(Node):
    """夹爪控制服务"""
    def __init__(self):
        super().__init__('gripper_service')
        self._gripper_open = False

    def handle_open(self, req, resp):
        self._gripper_open = True
        resp.success = True
        resp.message = "夹爪已打开"
        self.get_logger().info(resp.message)
        return resp

    def handle_close(self, req, resp):
        self._gripper_open = False
        resp.success = True
        resp.message = "夹爪已关闭"
        self.get_logger().info(resp.message)
        return resp

gripper_svc = GripperService()
resp1 = gripper_svc.handle_open(MockRequest(), MockResponse())
resp2 = gripper_svc.handle_close(MockRequest(), MockResponse())
print(f"  结果1: {resp1}")
print(f"  结果2: {resp2}")
print()

# ===== 4. Action 模式（Mock） =====
print("=" * 60)
print("4. Action（目标 / 反馈 / 结果）")
print("=" * 60)

import math

class NavigationAction(Node):
    """模拟 Nav2 导航动作服务端逻辑"""
    def __init__(self):
        super().__init__('navigation_action')

    def execute(self, goal_x, goal_y, current_x=0.0, current_y=0.0):
        self.get_logger().info(
            f'收到导航目标: ({goal_x:.2f}, {goal_y:.2f})'
        )
        total_dist = math.hypot(goal_x - current_x, goal_y - current_y)
        steps = 5
        for i in range(1, steps + 1):
            # 线性插值模拟移动
            t = i / steps
            cx = current_x + t * (goal_x - current_x)
            cy = current_y + t * (goal_y - current_y)
            remaining = math.hypot(goal_x - cx, goal_y - cy)
            self.get_logger().info(
                f'  [反馈] 进度 {100*t:.0f}% — 当前({cx:.2f},{cy:.2f}) — 剩余 {remaining:.2f}m'
            )
        self.get_logger().info('✅ 导航完成！')
        return {"success": True, "final_pose": (goal_x, goal_y)}

nav = NavigationAction()
result = nav.execute(3.0, 2.0)
print(f"  最终结果: {result}")
print()

# ===== 5. TF2 坐标变换（模拟） =====
print("=" * 60)
print("5. TF2 坐标变换（模拟）")
print("=" * 60)

import numpy as np

def rotation_matrix_z(theta):
    """绕 Z 轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def transform_point(point, translation, rotation_theta):
    """将 point 从局部坐标系变换到世界坐标系"""
    R = rotation_matrix_z(rotation_theta)
    return R @ np.array(point) + np.array(translation)

# 机器人底盘在 map 系的位置和朝向
robot_pos = [2.0, 1.0, 0.0]       # (x, y, z)
robot_yaw = np.deg2rad(45)          # 偏航角 45°

# 末端执行器在 base_link 系的局部坐标
ee_local = [0.5, 0.0, 0.3]

# 变换到 map 系
ee_global = transform_point(ee_local, robot_pos, robot_yaw)
print(f"  机器人位置(map系): {robot_pos}")
print(f"  末端执行器(base_link系): {ee_local}")
print(f"  末端执行器(map系): [{ee_global[0]:.3f}, {ee_global[1]:.3f}, {ee_global[2]:.3f}]")
print()

# ===== 6. 完整工作空间与包结构 =====
print("=" * 60)
print("6. ROS 2 工程结构与常用命令")
print("=" * 60)

workspace_structure = """
ros2_ws/                         ← 工作空间根目录
├── src/                         ← 源码目录
│   └── my_vla_pkg/              ← 功能包
│       ├── package.xml          ← 包描述（依赖声明）
│       ├── setup.py             ← Python 包配置
│       ├── setup.cfg
│       ├── my_vla_pkg/
│       │   ├── __init__.py
│       │   ├── vla_node.py      ← VLA 控制节点
│       │   └── utils.py
│       ├── launch/
│       │   └── vla.launch.py    ← 启动文件
│       ├── config/
│       │   └── params.yaml      ← 参数文件
│       └── test/
├── install/                     ← 构建产物（source 后可用）
├── build/
└── log/
"""
print(workspace_structure)

commands = {
    "构建工作空间":           "colcon build --symlink-install",
    "激活工作空间":            "source install/setup.bash",
    "创建新功能包":            "ros2 pkg create --build-type ament_python my_pkg",
    "运行节点":               "ros2 run my_vla_pkg vla_node",
    "启动 launch 文件":       "ros2 launch my_vla_pkg vla.launch.py",
    "查看话题列表":            "ros2 topic list",
    "查看节点计算图":          "rqt_graph",
    "录制 bag":               "ros2 bag record -a -o my_bag",
    "回放 bag":               "ros2 bag play my_bag",
}

print("  常用命令速查:")
for desc, cmd in commands.items():
    print(f"  {'':2}{desc:20s}: {cmd}")
print()

# ===== 7. params.yaml 示例 =====
print("=" * 60)
print("7. 参数文件示例 (config/params.yaml)")
print("=" * 60)

params_yaml = """
vla_control:
  ros__parameters:
    model_name: "openvla/openvla-7b"
    control_rate: 5.0
    action_scale: 0.05
    debug_mode: false
    joint_names:
      - "shoulder_pan_joint"
      - "shoulder_lift_joint"
      - "elbow_joint"
      - "wrist_1_joint"
      - "wrist_2_joint"
      - "wrist_3_joint"
"""
print(params_yaml)
print("  加载方式:")
print("    ros2 run my_pkg vla_node --ros-args --params-file config/params.yaml")
print("    ros2 launch my_pkg vla.launch.py  # 在 launch 文件中用 parameters=[...] 指定")
```

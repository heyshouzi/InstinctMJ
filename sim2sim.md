# Sim-to-Sim 观测系统详解

---

## 1. Policy 观测顺序

Policy 观测顺序 (字典插入顺序)

| 序号 |     观测名称      |            函数             | Shape (单帧) | 历史长度 | Scale | 坐标 Frame |
|------|-------------------|-----------------------------|--------------|----------|-------|------------|
| 1    | base_ang_vel      | envs_mdp.base_ang_vel       | (3,)         | 8        | 0.25  | Base (b)   |
| 2    | projected_gravity | envs_mdp.projected_gravity  | (3,)         | 8        | -     | Base (b)   |
| 3    | velocity_commands | envs_mdp.generated_commands | (3,)         | 8        | -     | Base (b)   |
| 4    | joint_pos         | envs_mdp.joint_pos_rel      | (31,)        | 8        | -     | Joint Space |
| 5    | joint_vel         | envs_mdp.joint_vel_rel      | (31,)        | 8        | 0.05  | Joint Space |
| 6    | actions           | envs_mdp.last_action        | (31,)        | 8        | -     | Action Space |
| 7    | depth_image       | delayed_visualizable_image  | (8, 18, 32)  | -        | -     | Camera      |

---

## 2. 坐标 Frame 说明

| 坐标 Frame | 说明 |
|-----------|------|
| **Base (b)** | 机器人本体坐标系 (body frame)。`base_ang_vel_b` 表示本体坐标系下的角速度。 |
| **Joint Space** | 关节空间。`joint_pos_rel` 为相对于默认关节位置的偏移量。 |
| **Action Space** | 动作空间。`actions` 为上一帧执行的动作。 |
| **Camera** | 相机坐标系。`depth_image` 为相机拍摄的深度图。 |

**Base (b) 坐标系定义:**
- X 轴: 机器人前进方向
- Y 轴: 机器人左侧方向
- Z 轴: 垂直向上

---

## 3. 历史写入 Buffer 的关系

### 3.1 非视觉观测 (joint_pos, joint_vel, base_ang_vel 等)

这些观测由 mjlab 的观察系统管理：

每次环境步进:
1. `ObservationTerm.__call__()` 被调用
2. 读取当前观测值
3. 写入该 Term 的历史缓冲区 (ring buffer)
4. 返回: 当前值 + 历史值 (共 history_length 帧)
5. 应用 noise (如果配置了)
6. 应用 scale (如果有)
7. flatten (如果 flatten_history_dim=True)

对于 flatten_history_dim=True 的观测:
- 原始 shape: `(history_length, D)`
- flatten 后 shape: `(history_length × D,)`

### 3.1.1 历史 Buffer 顺序: 最旧 → 最新

Ring buffer 存储顺序为 **[最旧, ..., 最新]**，flatten 时按 axis=0 展平：

```
Ring Buffer (history_length=8):
[t-7, t-6, t-5, t-4, t-3, t-2, t-1, t]
   0     1     2     3     4     5     6     7
   ↑                                ↑
最旧                                最新

Flatten 后 (按 axis=0):
[t-7_0, t-7_1, ..., t-7_D, t-6_0, ..., t_0, t_1, ...]
   0      1         D    D+1         D*7     D*7+D-1
   ↑
最旧 (index 0)
```

**所以:**
- **index 0 ~ D-1**: t-7 (最旧帧)
- **index D*7 ~ D*8-1**: t (最新帧)

示例 (base_ang_vel, D=3):
- flatten 后: `[ωx_t-7, ωy_t-7, ωz_t-7, ωx_t-6, ..., ωx_t, ωy_t, ωz_t]`
- shape: (24,) = 8帧 × 3维

### 3.2 视觉观测 (depth_image) - 特殊处理

depth_image 使用**独立的历史缓冲系统**，不依赖 mjlab 的观察历史。

---

## 3.3 深度观测处理详细流程

### 3.3.1 相机传感器配置

```python
NoisyGroupedRayCasterCameraCfg(
    data_types=["distance_to_image_plane"],
    noise_pipeline={
        "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
        "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
        "depth_normalization": DepthNormalizationCfg(
            depth_range=(0.0, 2.5),
            normalize=True,
            output_range=(0.0, 1.0),
        ),
    },
    data_histories={"distance_to_image_plane_noised": 37},
    min_distance=0.1,
    max_distance=2.5,
)
```

### 3.3.2 深度观测处理流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         深度观测处理完整流程                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

每次环境步进:
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Ray Caster 计算 (GroupedRayCasterCamera.postprocess_rays())              │
│   - 投射射线获取深度: distance_to_image_plane (原始深度)                           │
│   - 输出 shape: (N, H, W, 1)                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Noise Pipeline (NoisyCameraMixin.apply_noise_pipeline())                   │
│   按顺序应用以下噪声:                                                               │
│                                                                                     │
│   2.1 CropAndResize: 裁剪并缩放                                                     │
│       - crop_region: (18, 0, 16, 16) → 从 (16, 16) 裁剪区域                      │
│       - 输出: (N, 16, 16, 1)                                                      │
│                                                                                     │
│   2.2 GaussianBlur: 高斯模糊                                                        │
│       - kernel_size: 3, sigma: 1                                                   │
│       - 输出: (N, 16, 16, 1)                                                      │
│                                                                                     │
│   2.3 DepthNormalization: 深度归一化                                                │
│       - depth_range: (0.0, 2.5) → 裁剪到有效范围                                  │
│       - normalize: True → (d - 0.0) / (2.5 - 0.0)                                │
│       - output_range: (0.0, 1.0) → 线性映射到 [0, 1]                              │
│       - 输出: (N, 16, 16, 1), 值域 [0.0, 1.0]                                    │
│                                                                                     │
│   最终输出: distance_to_image_plane_noised (N, 16, 16, 1)                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Step 3: 写入历史缓冲区 (NoisyCameraMixin.update_history_buffers())                  │
│   - AsyncCircularBuffer(history_length=37)                                         │
│   - 每次步进 append 新帧                                                           │
│   - 内部存储: distance_to_image_plane_noised_history (N, 37, 16, 16, 1)          │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Step 4: Policy 观测采样 (delayed_visualizable_image.__call__())                    │
│                                                                                     │
│   配置参数:                                                                         │
│   - data_type: "distance_to_image_plane_noised_history"                           │
│   - history_skip_frames: 5 (每隔5帧采样一次)                                       │
│   - num_output_frames: 8 (输出8帧)                                                 │
│   - delayed_frame_ranges: (0, 1) (随机延迟0~1帧)                                    │
│                                                                                     │
│   采样逻辑:                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────┐ │
│   │ Buffer 索引计算:                                                              │ │
│   │ frame_offset = [35, 30, 25, 20, 15, 10, 5, 0]  (flip 后的结果)             │ │
│   │                                                                               │ │
│   │ frame_indices[n, f] = sensor_history_length                                  │ │
│   │                         - frame_offset[f]                                    │ │
│   │                         - num_delayed_frames[n]                              │ │
│   │                         - 1                                                  │ │
│   │                                                                               │ │
│   │ 其中 num_delayed_frames[n] ∈ [0, 1] (均匀分布)                             │ │
│   └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│   最终输出: delayed_frames (N, 8, 18, 32)                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3.3 各步骤代码位置

| 步骤 | 代码文件 | 类/函数 | 关键操作 |
|------|---------|---------|----------|
| 1 | `sensors/grouped_ray_caster.py` | `GroupedRayCasterCamera.postprocess_rays()` | 射线投射计算深度 |
| 2 | `sensors/noisy_camera/noisy_camera.py` | `NoisyCameraMixin.apply_noise_pipeline()` | 顺序应用噪声 |
| 2.1 | `utils/noise/noise_model.py` | `crop_and_resize()` | 裁剪缩放 (18,0,16,16) |
| 2.2 | `utils/noise/noise_model.py` | `gaussian_blur_noise()` | 高斯模糊 k=3, σ=1 |
| 2.3 | `utils/noise/noise_model.py` | `depth_normalization()` | 归一化到 [0,1] |
| 3 | `sensors/noisy_camera/noisy_camera.py` | `NoisyCameraMixin.update_history_buffers()` | 写入 AsyncCircularBuffer |
| 4 | `envs/mdp/observations/exteroception.py` | `delayed_visualizable_image.__call__()` | 采样延迟帧 |

### 3.3.4 数据形状变化汇总

| 阶段 | 数据名称 | Shape | 说明 |
|------|----------|-------|------|
| 原始 | `distance_to_image_plane` | (N, H, W, 1) | 射线投射原始深度 |
| 裁剪后 | `cropped` | (N, 16, 16, 1) | 裁剪到 16×16 |
| 噪声后 | `distance_to_image_plane_noised` | (N, 16, 16, 1) | 高斯模糊+归一化 |
| 历史缓冲 | `distance_to_image_plane_noised_history` | (N, 37, 16, 16, 1) | 37帧历史 |
| Policy输出 | `delayed_frames` | (N, 8, 18, 32) | 8帧降采样输出 |

### 3.3.5 delayed_visualizable_image 核心代码解析

```python
class delayed_visualizable_image(ManagerTermBase):
    def __init__(self, cfg, env):
        # ...
        self.history_skip_frames = 5        # 每隔5帧采样
        self.num_output_frames = 8          # 输出8帧
        self.delayed_frame_ranges = (0, 1) # 随机延迟0~1帧

        # frame_offset = [35, 30, 25, 20, 15, 10, 5, 0]
        # flip 后从最大间隔开始
        self.frame_offset = torch.flip(
            torch.arange(0, 40, 5), dims=(0,)
        )

    def __call__(self, ...):
        # 获取历史数据: (N, 37, 16, 16, 1)
        images = self.sensor.data.output[self.data_type]

        # 计算帧索引
        frame_indices = (
            37                      # sensor_history_length
            - self.frame_offset    # [35, 30, 25, 20, 15, 10, 5, 0]
            - self._num_delayed_frames  # [0, 1] 随机延迟
            - 1
        )  # → (N, 8), 每列是最远到最近的8帧索引

        # 高级索引采样
        batch_indices = torch.arange(N).unsqueeze(1).expand(-1, 8)
        delayed_frames = images[batch_indices, frame_indices]  # (N, 8, 18, 32)

        return delayed_frames
```

### 3.3.6 噪声 Pipeline 详细说明

| 噪声类型 | 配置 | 作用 |
|---------|------|------|
| **CropAndResize** | crop_region=(18, 0, 16, 16) | 从 36×64 裁剪出 18×32 区域 (高×宽) |
| **GaussianBlur** | kernel_size=3, sigma=1 | 3×3 高斯核模糊，模拟相机散焦 |
| **DepthNormalization** | depth_range=(0, 2.5), output_range=(0, 1) | 裁剪到 2.5m 并归一化到 [0,1] |

---

## 4. 哪些量做了 Scale

|   观测名称   | Scale 值 |       说明       |
|-------------|----------|-----------------|
| base_ang_vel | 0.25     | 角速度缩小4倍    |
| joint_vel    | 0.05     | 关节速度缩小20倍 |

Scale 应用位置: 在观测值返回之前，scale 参数直接与观测值相乘：
```
output = observation_value * scale
```

注意: scale 与 noise 是独立的，可以同时配置。

---



## 6. 观测处理流程图

### Policy 观测计算流程

每帧:

#### 1. 普通观测 (base_ang_vel, joint_pos 等):

```
ObservationTerm.__call__()
  - 获取当前观测值: (D,)
  - 从历史缓冲区读取过去 7 帧
  - 拼接: (8, D)
  - 应用 noise: (8, D) ± noise
  - 应用 scale: (8, D) × scale (仅 base_ang_vel, joint_vel)
  - flatten: (8*D,)
  - 返回: (8*D,) 或 (8, D) 如果 flatten_history_dim=False
```

#### 2. 深度观测 (depth_image):

```
delayed_visualizable_image.__call__()
  - 从 AsyncCircularBuffer 读取 distance_to_image_plane_noised_history: (N, 37, 18, 32, 1)
  - 根据 delay 采样帧索引
  - 根据 skip=5 降采样
  - 输出: (N, 8, 18, 32)
  - 返回: (N, 8, 18, 32)
```

最终返回 (如果 concatenate_terms=False):
```python
{
    "base_ang_vel": (N, 24),
    "projected_gravity": (N, 24),
    "velocity_commands": (N, 24),
    "joint_pos": (N, 248),
    "joint_vel": (N, 248),
    "actions": (N, 248),
    "depth_image": (N, 8, 18, 32),
}
```

---

## 7. 关键配置参数汇总

|         参数         |   值    |        说明        |
|---------------------|---------|-------------------|
| history_length       | 8       | 普通观测的历史帧数 |
| flatten_history_dim  | True    | 是否将历史展平     |
| scale (base_ang_vel) | 0.25    | 角速度缩放因子     |
| scale (joint_vel)    | 0.05    | 关节速度缩放因子   |
| noise                | Uniform | 均匀噪声           |
| depth_image skip     | 5       | 每隔5帧采样        |
| depth_image frames   | 8       | 输出8帧            |
| depth_image delay    | 0~1     | 随机延迟0-1帧      |
| depth_image buffer   | 37      | 相机内部缓冲37帧   |

---

## 8. 关节 PD 控制参数 (T2_v3)

### 8.1 PD 控制公式

```
KP = armature × NATURAL_FREQ²
KD = 2.0 × DAMPING_RATIO × armature × NATURAL_FREQ
ACTION_SCALE = 0.25 × effort_limit / KP
```

其中:
- `NATURAL_FREQ = 10 × 2π = 62.83 rad/s`
- `DAMPING_RATIO = 2.0`

### 8.2 执行器类型参数表

| 执行器 | 电机型号 | Armature (kg·m²) | KP | KD | Effort Limit (Nm) | Action Scale |
|--------|---------|-----------------|-----|------|-------------------|--------------|
| LING14 | 灵足14Nm | 0.00100 | 3.95 | 0.25 | 14.0 | 0.886 |
| 7025 | 自研7025 | 0.02782 | 109.82 | 3.50 | 107.0 | 0.244 |
| 5025 | 自研5025 | 0.00710 | 28.03 | 0.89 | 35.0 | 0.312 |
| 5036 | 高擎5036-02 | 0.01322 | 52.19 | 1.66 | 20.0 | 0.096 |
| 140NMB | 自研140NmB | 0.03073 | 121.32 | 3.86 | 130.0 | 0.268 |
| 4315 | 因克斯4315 | 0.03396 | 134.05 | 4.27 | 75.0 | 0.140 |

### 8.3 关节顺序表 (31 DOF)

| 序号 | 关节名称 | 执行器 | KP | KD | Action Scale |
|------|---------|--------|-----|------|--------------|
| 0 | aa_head_yaw_joint | LING14 | 3.95 | 0.25 | 0.886 |
| 1 | head_pitch_joint | LING14 | 3.95 | 0.25 | 0.886 |
| 2 | left_shoulder_pitch_joint | 7025 | 109.82 | 3.50 | 0.244 |
| 3 | left_shoulder_roll_joint | 7025 | 109.82 | 3.50 | 0.244 |
| 4 | left_elbow_pitch_joint | 5025 | 28.03 | 0.89 | 0.312 |
| 5 | left_elbow_yaw_joint | 5025 | 28.03 | 0.89 | 0.312 |
| 6 | left_wrist_pitch_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 7 | left_wrist_yaw_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 8 | left_wrist_roll_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 9 | right_shoulder_pitch_joint | 7025 | 109.82 | 3.50 | 0.244 |
| 10 | right_shoulder_roll_joint | 7025 | 109.82 | 3.50 | 0.244 |
| 11 | right_elbow_pitch_joint | 5025 | 28.03 | 0.89 | 0.312 |
| 12 | right_elbow_yaw_joint | 5025 | 28.03 | 0.89 | 0.312 |
| 13 | right_wrist_pitch_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 14 | right_wrist_yaw_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 15 | right_wrist_roll_joint | 5036 | 52.19 | 1.66 | 0.096 |
| 16 | waist_pitch_joint | 4315 | 134.05 | 4.27 | 0.140 |
| 17 | waist_roll_joint | 4315 | 134.05 | 4.27 | 0.140 |
| 18 | waist_yaw_joint | 7025 | 109.82 | 3.50 | 0.244 |
| 19 | left_hip_pitch_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 20 | left_hip_roll_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 21 | left_hip_yaw_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 22 | left_knee_pitch_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 23 | left_ankle_pitch_joint | 4315 | 134.05 | 4.27 | 0.140 |
| 24 | left_ankle_roll_joint | 4315 | 134.05 | 4.27 | 0.140 |
| 25 | right_hip_pitch_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 26 | right_hip_roll_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 27 | right_hip_yaw_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 28 | right_knee_pitch_joint | 140NMB | 121.32 | 3.86 | 0.268 |
| 29 | right_ankle_pitch_joint | 4315 | 134.05 | 4.27 | 0.140 |
| 30 | right_ankle_roll_joint | 4315 | 134.05 | 4.27 | 0.140 |

### 8.4 默认关节位置

机器人初始状态下的默认关节位置 (单位: rad)：

| 关节名称 | 默认位置 (rad) |
|---------|--------------|
| aa_head_yaw_joint | 0.00 |
| head_pitch_joint | 0.00 |
| left_shoulder_pitch_joint | 0.00 |
| left_shoulder_roll_joint | -1.40 |
| left_elbow_pitch_joint | 0.00 |
| left_elbow_yaw_joint | 0.00 |
| left_wrist_pitch_joint | 0.00 |
| left_wrist_yaw_joint | 0.00 |
| left_wrist_roll_joint | 0.00 |
| right_shoulder_pitch_joint | 0.00 |
| right_shoulder_roll_joint | 1.40 |
| right_elbow_pitch_joint | 0.00 |
| right_elbow_yaw_joint | 0.00 |
| right_wrist_pitch_joint | 0.00 |
| right_wrist_yaw_joint | 0.00 |
| right_wrist_roll_joint | 0.00 |
| waist_pitch_joint | 0.00 |
| waist_roll_joint | 0.00 |
| waist_yaw_joint | 0.00 |
| left_hip_pitch_joint | -0.20 |
| left_hip_roll_joint | 0.00 |
| left_hip_yaw_joint | 0.00 |
| left_knee_pitch_joint | 0.40 |
| left_ankle_pitch_joint | -0.20 |
| left_ankle_roll_joint | 0.00 |
| right_hip_pitch_joint | -0.20 |
| right_hip_roll_joint | 0.00 |
| right_hip_yaw_joint | 0.00 |
| right_knee_pitch_joint | 0.40 |
| right_ankle_pitch_joint | -0.20 |
| right_ankle_roll_joint | 0.00 |

机器人根部 (trunk) 位置: `(0.0, 0.0, 1.1)` m

---

## 9. T2_v3 MJCF 文件 (src/instinct_mj/assets/resources/booster_t2/T2_v3.xml)

```xml
<mujoco model="T2_v3">
  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
  <option timestep="0.001" tolerance="1e-6" impratio="10" solver="Newton"/>
  <default>
    <default class="T2">
      <default class="visual">
        <geom group="1" type="mesh" density="0" contype="0" conaffinity="0" rgba="0.75294 0.75294 0.75294 1"/>
      </default>
      <default class="collision">
        <geom group="3" contype="1" conaffinity="1" rgba="0.4549 0.7294 0.9569 0.9"/>
      </default>
    </default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    <!-- Visual meshes from URDF -->
    <mesh name="trunk" content_type="model/stl" file="trunk.STL"/>
    <mesh name="head_yaw_link" content_type="model/stl" file="head_yaw_link.STL"/>
    <mesh name="head_pitch_link" content_type="model/stl" file="head_pitch_link.STL"/>
    <mesh name="head_pitch_link_collision" content_type="model/stl" file="head_pitch_link_collision.STL"/>
    <mesh name="left_shoulder_pitch_link" content_type="model/stl" file="left_shoulder_pitch_link.STL"/>
    <mesh name="left_shoulder_roll_link" content_type="model/stl" file="left_shoulder_roll_link.STL"/>
    <mesh name="left_elbow_pitch_link" content_type="model/stl" file="left_elbow_pitch_link.STL"/>
    <mesh name="left_elbow_yaw_link" content_type="model/stl" file="left_elbow_yaw_link.STL"/>
    <mesh name="left_wrist_pitch_link" content_type="model/stl" file="left_wrist_pitch_link.STL"/>
    <mesh name="left_wrist_yaw_link" content_type="model/stl" file="left_wrist_yaw_link.STL"/>
    <mesh name="left_wrist_roll_link" content_type="model/stl" file="left_wrist_roll_link.STL"/>
    <mesh name="left_wrist_roll_link_collision" content_type="model/stl" file="left_wrist_roll_link_collision.STL"/>
    <mesh name="right_shoulder_pitch_link" content_type="model/stl" file="right_shoulder_pitch_link.STL"/>
    <mesh name="right_shoulder_roll_link" content_type="model/stl" file="right_shoulder_roll_link.STL"/>
    <mesh name="right_elbow_pitch_link" content_type="model/stl" file="right_elbow_pitch_link.STL"/>
    <mesh name="right_elbow_yaw_link" content_type="model/stl" file="right_elbow_yaw_link.STL"/>
    <mesh name="right_wrist_pitch_link" content_type="model/stl" file="right_wrist_pitch_link.STL"/>
    <mesh name="right_wrist_yaw_link" content_type="model/stl" file="right_wrist_yaw_link.STL"/>
    <mesh name="right_wrist_roll_link" content_type="model/stl" file="right_wrist_roll_link.STL"/>
    <mesh name="right_wrist_roll_link_collision" content_type="model/stl" file="right_wrist_roll_link_collision.STL"/>
    <mesh name="waist_pitch_link" content_type="model/stl" file="waist_pitch_link.STL"/>
    <mesh name="waist_roll_link" content_type="model/stl" file="waist_roll_link.STL"/>
    <mesh name="waist_yaw_link" content_type="model/stl" file="waist_yaw_link.STL"/>
    <mesh name="waist_yaw_link_collision" content_type="model/stl" file="waist_yaw_link_collision.STL"/>
    <mesh name="left_hip_pitch_link" content_type="model/stl" file="left_hip_pitch_link.STL"/>
    <mesh name="left_hip_roll_link" content_type="model/stl" file="left_hip_roll_link.STL"/>
    <mesh name="left_hip_yaw_link" content_type="model/stl" file="left_hip_yaw_link.STL"/>
    <mesh name="left_knee_pitch_link" content_type="model/stl" file="left_knee_pitch_link.STL"/>
    <mesh name="left_ankle_pitch_link" content_type="model/stl" file="left_ankle_pitch_link.STL"/>
    <mesh name="left_ankle_roll_link" content_type="model/stl" file="left_ankle_roll_link.STL"/>
    <mesh name="right_hip_pitch_link" content_type="model/stl" file="right_hip_pitch_link.STL"/>
    <mesh name="right_hip_roll_link" content_type="model/stl" file="right_hip_roll_link.STL"/>
    <mesh name="right_hip_yaw_link" content_type="model/stl" file="right_hip_yaw_link.STL"/>
    <mesh name="right_knee_pitch_link" content_type="model/stl" file="right_knee_pitch_link.STL"/>
    <mesh name="right_ankle_pitch_link" content_type="model/stl" file="right_ankle_pitch_link.STL"/>
    <mesh name="right_ankle_roll_link" content_type="model/stl" file="right_ankle_roll_link.STL"/>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 10" dir="0 0 -1"/>
    <!-- Trunk with free joint to world -->
    <body name="trunk" pos="0 0 0">
      <inertial pos="0.051147 0.000498 0.078988" mass="10.4714" diaginertia="0.104144 0.076916 0.055820"/>
      <site name='imu' size='0.01' pos='0.0 0.0 0.0'/>
      <joint name="world_joint" type="free"/>
      <geom class="visual" mesh="trunk"/>
      <geom name="trunk_collision" class="collision" size="0.09 0.1 0.125" pos="0.05 0 0.1" type="box"/>

      <!-- Head chain: trunk -> head_yaw -> head_pitch -->
      <body name="head_yaw_link" pos="0.0502 0 0.2044">
        <inertial pos="0.000106 -0.000532 0.088591" mass="0.554359" diaginertia="0.000759 0.000764 0.000230"/>
        <joint name="aa_head_yaw_joint" pos="0 0 0" axis="0 0 1" range="-1.04712 1.04712" actuatorfrcrange="-12 12" armature="0.001"/>
        <geom class="visual" mesh="head_yaw_link"/>
        <body name="head_pitch_link" pos="0 0 0.10627">
          <inertial pos="0.03176 0.00059332 0.052826" mass="0.80336" diaginertia="0.003221 0.003718 0.003238"/>
          <joint name="head_pitch_joint" pos="0 0 0" axis="0 1 0" range="-0.34904 0.4363" actuatorfrcrange="-12 12" armature="0.001"/>
          <geom class="visual" mesh="head_pitch_link"/>
          <geom name="head_collision" class="collision" type="mesh" mesh="head_pitch_link_collision" contype="1" conaffinity="1"/>
        </body>
      </body>

      <!-- Left arm: trunk -> shoulder_pitch -> shoulder_roll -> elbow_pitch -> elbow_yaw -> wrist_pitch -> wrist_yaw -> wrist_roll -->
      <body name="left_shoulder_pitch_link" pos="0.05 0.11571 0.16121" quat="0.997564 0.0697583 0 0">
        <inertial pos="0.006298 0.043315 -0.009687" mass="0.568750" diaginertia="0.000284 0.000296 0.000322"/>
        <joint name="left_shoulder_pitch_joint" pos="0 0 0" axis="0 1 0" range="-3.2289 1.7453" actuatorfrcrange="-80 80" armature="0.02781875"/>
        <geom class="visual" mesh="left_shoulder_pitch_link"/>
        <body name="left_shoulder_roll_link" pos="0.003 0.045407 -0.01" quat="0.997564 -0.0697583 0 0">
          <inertial pos="0.001444 0.058404 -0.000002" mass="0.753752" diaginertia="0.000660 0.000470 0.000758"/>
          <joint name="left_shoulder_roll_joint" pos="0 0 0" axis="1 0 0" range="-1.55323 1.55323" actuatorfrcrange="-30 30" armature="0.0071"/>
          <geom class="visual" mesh="left_shoulder_roll_link"/>
          <body name="left_elbow_pitch_link" pos="0 0.1 0">
            <inertial pos="0.003057 0.090796 0.001021" mass="0.785860" diaginertia="0.001285 0.000444 0.001225"/>
            <joint name="left_elbow_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.0944 2.0944" actuatorfrcrange="-30 30" armature="0.0071"/>
            <geom class="visual" mesh="left_elbow_pitch_link"/>
            <geom name="left_upperarm_collision" class="collision" size="0.03 0.075 0.03" pos="0 -0.04 0" type="box"/>
            <body name="left_elbow_yaw_link" pos="0.005 0.11 0">
              <inertial pos="-0.000546 0.05742 -0.000827" mass="0.497020" diaginertia="0.000401 0.000209 0.000318"/>
              <joint name="left_elbow_yaw_joint" pos="0 0 0" axis="0 0 1" range="-2.138 0" actuatorfrcrange="-30 30" armature="0.0071"/>
              <geom class="visual" mesh="left_elbow_yaw_link"/>
              <body name="left_wrist_pitch_link" pos="0 0.0904 0">
                <inertial pos="-0.000116 0.048621 0.000981" mass="0.661340" diaginertia="0.000614 0.000371 0.000470"/>
                <joint name="left_wrist_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.0943 2.0943" actuatorfrcrange="-16 16" armature="0.0132192"/>
                <geom class="visual" mesh="left_wrist_pitch_link"/>
                <geom name="left_forearm_collision" class="collision" size="0.03 0.06 0.03" type="box"/>
                <body name="left_wrist_yaw_link" pos="0 0.1196 0">
                  <inertial pos="0.001506 -0.00167 -0.000632" mass="0.422577" diaginertia="0.000223 0.000220 0.000230"/>
                  <joint name="left_wrist_yaw_joint" pos="0 0 0" axis="0 0 1" range="-0.7836 0.7138" actuatorfrcrange="-16 16" armature="0.0132192"/>
                  <geom class="visual" mesh="left_wrist_yaw_link"/>
                  <body name="left_wrist_roll_link">
                    <inertial pos="0.008307 0.101688 -0.002439" mass="0.534408" diaginertia="0.001923 0.000668 0.001981"/>
                    <joint name="left_wrist_roll_joint" pos="0 0 0" axis="1 0 0" range="-1.3962 0.9599" actuatorfrcrange="-16 16" armature="0.0132192"/>
                    <geom class="visual" mesh="left_wrist_roll_link"/>
                    <geom name="left_hand_collision" class="collision" type="mesh" mesh="left_wrist_roll_link_collision"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Right arm: trunk -> shoulder_pitch -> shoulder_roll -> elbow_pitch -> elbow_yaw -> wrist_pitch -> wrist_yaw -> wrist_roll -->
      <body name="right_shoulder_pitch_link" pos="0.05 -0.11571 0.16121" quat="0.997564 -0.0697583 0 0">
        <inertial pos="0.006061 -0.043336 -0.00968" mass="0.568256" diaginertia="0.000285 0.000302 0.000329"/>
        <joint name="right_shoulder_pitch_joint" pos="0 0 0" axis="0 1 0" range="-3.2289 1.7453" actuatorfrcrange="-80 80" armature="0.02781875"/>
        <geom class="visual" mesh="right_shoulder_pitch_link"/>
        <body name="right_shoulder_roll_link" pos="0.003 -0.045407 -0.01" quat="0.997564 0.0697583 0 0">
          <inertial pos="0.001444 -0.058404 -0.000002" mass="0.753752" diaginertia="0.000660 0.000470 0.000758"/>
          <joint name="right_shoulder_roll_joint" pos="0 0 0" axis="1 0 0" range="-1.55323 1.55323" actuatorfrcrange="-30 30" armature="0.0071"/>
          <geom class="visual" mesh="right_shoulder_roll_link"/>
          <body name="right_elbow_pitch_link" pos="0 -0.1 0">
            <inertial pos="0.003057 -0.090796 0.001021" mass="0.785860" diaginertia="0.001285 0.000444 0.001225"/>
            <joint name="right_elbow_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.0944 2.0944" actuatorfrcrange="-30 30" armature="0.0071"/>
            <geom class="visual" mesh="right_elbow_pitch_link"/>
            <geom name="right_upperarm_collision" class="collision" size="0.03 0.075 0.03" pos="0 0.04 0" type="box"/>
            <body name="right_elbow_yaw_link" pos="0.005 -0.11 0">
              <inertial pos="-0.000546 -0.05742 -0.000827" mass="0.497020" diaginertia="0.000401 0.000209 0.000318"/>
              <joint name="right_elbow_yaw_joint" pos="0 0 0" axis="0 0 1" range="0 2.138" actuatorfrcrange="-30 30" armature="0.0071"/>
              <geom class="visual" mesh="right_elbow_yaw_link"/>
              <body name="right_wrist_pitch_link" pos="0 -0.0904 0">
                <inertial pos="-0.000061 -0.048591 0.000932" mass="0.653574" diaginertia="0.000601 0.000365 0.000459"/>
                <joint name="right_wrist_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.0943 2.0943" actuatorfrcrange="-16 16" armature="0.0132192"/>
                <geom class="visual" mesh="right_wrist_pitch_link"/>
                <geom name="right_forearm_collision" class="collision" size="0.03 0.06 0.03" type="box"/>
                <body name="right_wrist_yaw_link" pos="0 -0.1196 0">
                  <inertial pos="0.001511 0.00167 -0.000632" mass="0.420203" diaginertia="0.000223 0.000220 0.000230"/>
                  <joint name="right_wrist_yaw_joint" pos="0 0 0" axis="0 0 1" range="-0.7138 0.7836" actuatorfrcrange="-16 16" armature="0.0132192"/>
                  <geom class="visual" mesh="right_wrist_yaw_link"/>
                  <body name="right_wrist_roll_link">
                    <inertial pos="0.008285 -0.10667 -0.002341" mass="0.532410" diaginertia="0.001934 0.000656 0.002041"/>
                    <joint name="right_wrist_roll_joint" pos="0 0 0" axis="1 0 0" range="-0.9599 1.3962" actuatorfrcrange="-16 16" armature="0.0132192"/>
                    <geom class="visual" mesh="right_wrist_roll_link"/>
                    <geom name="right_hand_collision" class="collision" type="mesh" mesh="right_wrist_roll_link_collision"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Waist: trunk -> waist_pitch -> waist_roll -> waist_yaw -->
      <body name="waist_pitch_link" pos="0.0505 0 -0.0725">
        <inertial pos="0 0 -0.007902" mass="0.086922" diaginertia="0.000008 0.000015 0.000013"/>
        <joint name="waist_pitch_joint" pos="0 0 0" axis="0 1 0" range="-0.7853 0.5934" actuatorfrcrange="-60 60" armature="0.0339552"/>
        <geom class="visual" mesh="waist_pitch_link"/>
        <body name="waist_roll_link" pos="0 0 -0.01">
          <inertial pos="0.000493 -0.000031 -0.050659" mass="1.486622" diaginertia="0.002232 0.002330 0.001547"/>
          <joint name="waist_roll_joint" pos="0 0 0" axis="1 0 0" range="-0.4363 0.4363" actuatorfrcrange="-60 60" armature="0.0339552"/>
          <geom class="visual" mesh="waist_roll_link"/>
          <body name="waist_yaw_link" pos="0.004 0 -0.1026">
            <inertial pos="0.000379 -0.000304 -0.053490" mass="3.63622" diaginertia="0.010297 0.004545 0.009886"/>
            <joint name="waist_yaw_joint" pos="0 0 0" axis="0 0 1" range="-2.4434 2.4434" actuatorfrcrange="-107 107" armature="0.02781875"/>
            <geom class="visual" mesh="waist_yaw_link"/>
            <geom name="waist_collision" class="collision" type="mesh" mesh="waist_yaw_link_collision"/>

            <!-- Left leg: waist_yaw -> hip_pitch -> hip_roll -> hip_yaw -> knee_pitch -> ankle_pitch -> ankle_roll -->
            <body name="left_hip_pitch_link" pos="0 0.073144 -0.064256" quat="0.965926 -0.258819 0 0">
              <inertial pos="-0.010014 0.042267 -0.056268" mass="1.429169" diaginertia="0.001455 0.001219 0.001069"/>
              <joint name="left_hip_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.9670 2.2689" actuatorfrcrange="-105 105" armature="0.030729032"/>
              <geom class="visual" mesh="left_hip_pitch_link"/>
              <body name="left_hip_roll_link" pos="0 0.043604 -0.058926" quat="0.965926 0.258819 0 0">
                <inertial pos="0.015951 0.000371 -0.030987" mass="0.400274" diaginertia="0.000701 0.000657 0.000451"/>
                <joint name="left_hip_roll_joint" pos="0 0 0" axis="1 0 0" range="-1.2217 1.5708" actuatorfrcrange="-105 105" armature="0.030729032"/>
                <geom class="visual" mesh="left_hip_roll_link"/>
                <geom name="left_thigh_collision" class="collision" size="0.04 0.04 0.075" pos="0 0 -0.1" type="box"/>
                <body name="left_hip_yaw_link" pos="0 0 -0.0675">
                  <inertial pos="-0.013360 -0.004777 -0.115494" mass="3.052913" diaginertia="0.021867 0.022472 0.003893"/>
                  <joint name="left_hip_yaw_joint" pos="0 0 0" axis="0 0 1" range="-2.8012 2.8012" actuatorfrcrange="-60 60" armature="0.015069038"/>
                  <geom class="visual" mesh="left_hip_yaw_link"/>
                  <geom name="left_knee_collision" class="collision" size="0.05 0.055" pos="-0.03 -0.005 -0.2" quat="0.707107 0.707107 0 0" type="cylinder"/>
                  <body name="left_knee_pitch_link" pos="-0.03 0 -0.1972">
                    <inertial pos="0.019387 0.001973 -0.132921" mass="2.435191" diaginertia="0.014387 0.014743 0.002349"/>
                    <joint name="left_knee_pitch_joint" pos="0 0 0" axis="0 1 0" range="0 2.6179" actuatorfrcrange="-80 80" armature="0.030729032"/>
                    <geom class="visual" mesh="left_knee_pitch_link"/>
                    <geom name="left_calf_collision" class="collision" size="0.05 0.04 0.0875" pos="0.015 0 -0.17" type="box"/>
                    <body name="left_ankle_pitch_link" pos="0.033 0 -0.32835">
                      <inertial pos="0 0 -0.004791" mass="0.131833" diaginertia="0.000018 0.000017 0.000021"/>
                      <joint name="left_ankle_pitch_joint" pos="0 0 0" axis="0 1 0" range="-0.8604 0.6108" actuatorfrcrange="-50 50" armature="0.0339552"/>
                      <geom class="visual" mesh="left_ankle_pitch_link"/>
                      <body name="left_ankle_roll_link" pos="0 0 -0.01">
                        <inertial pos="-0.023375 0 -0.008981" mass="0.778635" diaginertia="0.000403 0.001806 0.001959"/>
                        <joint name="left_ankle_roll_joint" pos="0 0 0" axis="1 0 0" range="-0.3926 0.3926" actuatorfrcrange="-40 40" armature="0.0339552"/>
                        <geom class="visual" mesh="left_ankle_roll_link"/>
                        <geom name="left_ankle_collision" class="collision" type="mesh" mesh="left_ankle_roll_link"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>

            <!-- Right leg: waist_yaw -> hip_pitch -> hip_roll -> hip_yaw -> knee_pitch -> ankle_pitch -> ankle_roll -->
            <body name="right_hip_pitch_link" pos="0 -0.073144 -0.064256" quat="0.965926 0.258819 0 0">
              <inertial pos="-0.010014 -0.042267 -0.056268" mass="1.429169" diaginertia="0.001455 0.001219 0.001069"/>
              <joint name="right_hip_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.9670 2.2689" actuatorfrcrange="-105 105" armature="0.030729032"/>
              <geom class="visual" mesh="right_hip_pitch_link"/>
              <body name="right_hip_roll_link" pos="0 -0.043604 -0.058926" quat="0.965926 -0.258819 0 0">
                <inertial pos="0.015951 -0.000371 -0.030987" mass="0.400307" diaginertia="0.000701 0.000657 0.000451"/>
                <joint name="right_hip_roll_joint" pos="0 0 0" axis="1 0 0" range="-1.5708 1.2217" actuatorfrcrange="-105 105" armature="0.030729032"/>
                <geom class="visual" mesh="right_hip_roll_link"/>
                <geom name="right_thigh_collision" class="collision" size="0.04 0.04 0.075" pos="0 0 -0.1" type="box"/>
                <body name="right_hip_yaw_link" pos="0 0 -0.0675">
                  <inertial pos="-0.013412 0.004735 -0.115473" mass="3.052905" diaginertia="0.021857 0.022468 0.003900"/>
                  <joint name="right_hip_yaw_joint" pos="0 0 0" axis="0 0 1" range="-2.8012 2.8012" actuatorfrcrange="-60 60" armature="0.015069038"/>
                  <geom class="visual" mesh="right_hip_yaw_link"/>
                  <geom name="right_knee_collision" class="collision" size="0.05 0.055" pos="-0.03 0.005 -0.2" quat="0.707107 0.707107 0 0" type="cylinder"/>
                  <body name="right_knee_pitch_link" pos="-0.03 0 -0.1972">
                    <inertial pos="0.019475 -0.002558 -0.133908" mass="2.361510" diaginertia="0.014259 0.014614 0.002324"/>
                    <joint name="right_knee_pitch_joint" pos="0 0 0" axis="0 1 0" range="0 2.6179" actuatorfrcrange="-80 80" armature="0.030729032"/>
                    <geom class="visual" mesh="right_knee_pitch_link"/>
                    <geom name="right_calf_collision" class="collision" size="0.05 0.04 0.0875" pos="0.015 0 -0.17" type="box"/>
                    <body name="right_ankle_pitch_link" pos="0.033 0 -0.32835">
                      <inertial pos="0 0 -0.004791" mass="0.131833" diaginertia="0.000018 0.000017 0.000021"/>
                      <joint name="right_ankle_pitch_joint" pos="0 0 0" axis="0 1 0" range="-0.8604 0.6108" actuatorfrcrange="-50 50" armature="0.0339552"/>
                      <geom class="visual" mesh="right_ankle_pitch_link"/>
                      <body name="right_ankle_roll_link" pos="0 0 -0.01">
                        <inertial pos="0.023375 0 -0.008981" mass="0.778635" diaginertia="0.000403 0.001806 0.001959"/>
                        <joint name="right_ankle_roll_joint" pos="0 0 0" axis="1 0 0" range="-0.3926 0.3926" actuatorfrcrange="-40 40" armature="0.0339552"/>
                        <geom class="visual" mesh="right_ankle_roll_link"/>
                        <geom name="right_ankle_collision" class="collision" type="mesh" mesh="right_ankle_roll_link"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="aa_head_yaw_joint" joint="aa_head_yaw_joint" ctrlrange="-12.0 12.0" ctrllimited="true"/>
    <motor name="head_pitch_joint" joint="head_pitch_joint" ctrlrange="-12.0 12.0" ctrllimited="true"/>
    <motor name="left_shoulder_pitch_joint" joint="left_shoulder_pitch_joint" ctrlrange="-80.0 80.0" ctrllimited="true"/>
    <motor name="left_shoulder_roll_joint" joint="left_shoulder_roll_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="left_elbow_pitch_joint" joint="left_elbow_pitch_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="left_elbow_yaw_joint" joint="left_elbow_yaw_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="left_wrist_pitch_joint" joint="left_wrist_pitch_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="left_wrist_roll_joint" joint="left_wrist_roll_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint" ctrlrange="-80.0 80.0" ctrllimited="true"/>
    <motor name="right_shoulder_roll_joint" joint="right_shoulder_roll_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="right_elbow_pitch_joint" joint="right_elbow_pitch_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="right_elbow_yaw_joint" joint="right_elbow_yaw_joint" ctrlrange="-30.0 30.0" ctrllimited="true"/>
    <motor name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="right_wrist_roll_joint" joint="right_wrist_roll_joint" ctrlrange="-16.0 16.0" ctrllimited="true"/>
    <motor name="waist_pitch_joint" joint="waist_pitch_joint" ctrlrange="-60.0 60.0" ctrllimited="true"/>
    <motor name="waist_roll_joint" joint="waist_roll_joint" ctrlrange="-60.0 60.0" ctrllimited="true"/>
    <motor name="waist_yaw_joint" joint="waist_yaw_joint" ctrlrange="-107.0 107.0" ctrllimited="true"/>
    <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint" ctrlrange="-105.0 105.0" ctrllimited="true"/>
    <motor name="left_hip_roll_joint" joint="left_hip_roll_joint" ctrlrange="-105.0 105.0" ctrllimited="true"/>
    <motor name="left_hip_yaw_joint" joint="left_hip_yaw_joint" ctrlrange="-60.0 60.0" ctrllimited="true"/>
    <motor name="left_knee_pitch_joint" joint="left_knee_pitch_joint" ctrlrange="-80.0 80.0" ctrllimited="true"/>
    <motor name="left_ankle_pitch_joint" joint="left_ankle_pitch_joint" ctrlrange="-50.0 50.0" ctrllimited="true"/>
    <motor name="left_ankle_roll_joint" joint="left_ankle_roll_joint" ctrlrange="-40.0 40.0" ctrllimited="true"/>
    <motor name="right_hip_pitch_joint" joint="right_hip_pitch_joint" ctrlrange="-105.0 105.0" ctrllimited="true"/>
    <motor name="right_hip_roll_joint" joint="right_hip_roll_joint" ctrlrange="-105.0 105.0" ctrllimited="true"/>
    <motor name="right_hip_yaw_joint" joint="right_hip_yaw_joint" ctrlrange="-60.0 60.0" ctrllimited="true"/>
    <motor name="right_knee_pitch_joint" joint="right_knee_pitch_joint" ctrlrange="-80.0 80.0" ctrllimited="true"/>
    <motor name="right_ankle_pitch_joint" joint="right_ankle_pitch_joint" ctrlrange="-50.0 50.0" ctrllimited="true"/>
    <motor name="right_ankle_roll_joint" joint="right_ankle_roll_joint" ctrlrange="-40.0 40.0" ctrllimited="true"/>
  </actuator>
  <sensor>
    <framepos name="position" objtype="site" noise="0.001" objname="imu"/>
    <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>
    <velocimeter name="linear-velocity" site="imu" noise="0.001"/>
    <gyro name="angular-velocity" site="imu" noise="0.1"/>
  </sensor>
</mujoco>
```

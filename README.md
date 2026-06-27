# Piper-Sim2Real-RL: 基于领域随机化的 AgileX Piper 机械臂 6-DOF 目标追踪

本项目基于 **MuJoCo** 物理引擎与 **Stable-Baselines3 (SB3)** 强化学习框架，实现了 AgileX Piper 6自由度机械臂端到端（End-to-End）的空间目标追踪控制。项目重点攻克了仿真到真实世界迁移（Sim-to-Real）中的电机扰动难题，通过引入领域随机化（Domain Randomization）技术，使策略网络具备极强的鲁棒性。

---
## 效果演示与训练表现

### 1. 6-DOF 空间目标追踪表现 (Simulation Demo)
<p align="center">
  <img src="./docs/demo.gif" width="70%" alt="Piper Robot Tracking Demo" />
</p>

---
## 项目核心亮点与技术演进

### 1. 架构打桩：解决多进程向量环境的日志拦截 Bug
在自定义 Gym 环境初期，由于 SB3 的向量环境（`VecEnv`）会自动截断并隐式处理底层的 `terminated` 与 `truncated` 信号，导致终端日志中迟迟无法打印出 `rollout/ep_rew_mean`（平均回合回报），使训练处于“黑盒状态”。
* **解决方案**：放弃依赖易受干扰的物理引擎内部时间（`self.data.time`），在 Python 层面引入**显式步数计数器**（`self.current_step`），并在环境外层强制套上 `stable_baselines3.common.monitor.Monitor` 包装器进行底层数据打桩，成功量化了收敛指标。

### 2. 奖励机制：稠密奖励（Dense Reward）解耦探索困境
初期使用稀疏奖励时，机械臂在连续动作空间中盲目探索导致频繁训练超时。
* **解决方案**：重构 `step()` 函数，将奖励设计为与末端执行器（Link6）到目标点欧氏距离直接挂钩的高权重稠密距离惩罚：
  $$Reward = -3.0 \times \|Pos_{ee} - Pos_{target}\|_2$$
  配合 $+150.0$ 的巨额通关正奖励与动作平滑惩罚，完美引导网络快速开窍。

### 3. Sim-to-Real 鲁棒性：动态领域随机化（Domain Randomization）
为了对抗真实世界中电机间隙、发热衰减及摩擦力波动，项目主动制造物理“不完美”。
* **核心实现**：在每轮环境 `reset()` 时，随机生成一个 `0.6 ~ 1.4` 倍的电机出力干扰系数：
  ```python
  self.motor_efficiency = np.random.uniform(0.6, 1.4)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer  # 如果你用的被动渲染是旧版的，需要保留；如果是标准原生渲染可不写
import time

class PiperGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # 用于平滑惩罚，记录上一步的动作
        self.last_action = None

        # 1. 动作空间：6个关节速度/位置控制 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 2. 状态空间：6个关节角度 + 6个关节速度 + 3维末端位置 + 3维目标位置 = 18维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # 查找 Piper 末端夹爪的 ID
        try:
            self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        except:
            self.end_effector_id = 0 # 如果找不到，暂用基座代替
            
        # 初始目标点位置
        self.target_pos = np.array([0.2, 0.2, 0.3])
        
        # 【新增】在此处初始化默认的电机效率系数，防止 step 找不到它
        self.motor_efficiency = 1.0
        self.max_steps = 600  
        self.current_step = 0

    def _get_obs(self):
        ee_pos = self.data.xpos[self.end_effector_id]
        return np.concatenate([
            self.data.qpos[:6], 
            self.data.qvel[:6], 
            ee_pos, 
            self.target_pos
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        # 重置上一步动作为 None
        self.last_action = None
        
        # 【微调 1】缩小初始目标点的随机范围，让它更容易在机械臂的工作空间内，降低初期探索难度
        self.target_pos = np.random.uniform(low=[-0.2, 0.15, 0.15], high=[0.2, 0.35, 0.45])
        
        try:
            target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
            self.model.geom_pos[target_geom_id] = self.target_pos
        except:
            pass

        observation = self._get_obs()
        if self.render_mode == "human": self._render_frame()
        return observation, {}

    def step(self, action):
        #执行动作
        actual_action = action * 5.0 * self.motor_efficiency
        self.data.ctrl[:6] = actual_action
        # --------------------------------------------------------

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        ee_pos = self.data.xpos[self.end_effector_id] 

        # 计算距离
        distance = np.linalg.norm(ee_pos - self.target_pos)

        self.current_step += 1
        
        # 判定是否成功触碰
        terminated = False
        if distance < 0.05:  
            reward = 150.0  # 拿到大奖
            terminated = True
        else:
            reward = -3.0 * distance

        # 【核心修改】放弃使用 self.data.time，改用步数判定超时
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, truncated, {}
    

    def _render_frame(self):
        if self.viewer is None:
            # 启动 MuJoCo 的被动可视化窗口
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
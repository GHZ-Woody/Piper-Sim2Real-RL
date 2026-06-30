import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

class PiperAssemblyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.last_action = None
        self.current_step = 0
        self.max_steps = 600  

        # ====================================================================
        # 1. 查找 MuJoCo 中的关键物体 ID
        # ====================================================================
        # 机械臂末端夹爪 ID
        try:
            self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        except:
            self.end_effector_id = 0
            
        # 【新增】红插销与蓝基座（孔洞）的 ID
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_hole")

        # ====================================================================
        # 2. 动作与状态空间定义
        # ====================================================================
        # 动作空间：3维，代表末端夹爪在 3D 笛卡尔空间中的位移 [Δx, Δy, Δz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 状态空间（保持 21 维不变，因为眼睛看到的没变）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        # 查找我们的 mocap 目标点 ID
        self.mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")]

    def _get_obs(self):
        # 实时获取末端、插销、孔洞的 3D 绝对坐标
        ee_pos = self.data.xpos[self.end_effector_id]
        peg_pos = self.data.xpos[self.peg_body_id]
        hole_pos = self.data.xpos[self.hole_body_id]
        
        return np.concatenate([
            self.data.qpos[:6],    # 关节角度
            self.data.qvel[:6],    # 关节速度
            ee_pos,                # 末端位置
            peg_pos,               # 插销位置
            hole_pos               # 孔洞中心位置
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.last_action = None
        
        # 1. 恢复 mocap 向导点的初始安全位置 (让机械臂初始悬停在桌子上方)
        self.data.mocap_pos[self.mocap_id] = np.array([0.2, 0.0, 0.25])
        # 2. 随机初始化红插销的位置（保持不变）
        peg_x = np.random.uniform(0.26, 0.34)
        peg_y = np.random.uniform(0.06, 0.14)
        peg_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_joint")]
        self.data.qpos[peg_qpos_adr : peg_qpos_adr + 3] = [peg_x, peg_y, 0.2]

        # 刷新物理状态
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        if self.render_mode == "human": self._render_frame()
        return observation, {}

    def step(self, action):
        # 1. 将算法输出的 [-1, 1] 映射到实际物理位移，比如单步最大移动 1.5 厘米 (0.015 米)
        delta_pos = action * 0.015
        
        # 2. 核心：通过改变 mocap_pos 来移动向导点
        self.data.mocap_pos[self.mocap_id] += delta_pos
        
        # 3. 限制机械臂末端的探索边界，防止它沉入桌子底下或者飞得太远
        # X: [0.1, 0.5], Y: [-0.3, 0.3], Z: [0.11, 0.4] (0.11保证夹爪刚好贴着桌面 0.1, 留出1厘米间隙)
        self.data.mocap_pos[self.mocap_id] = np.clip(
            self.data.mocap_pos[self.mocap_id],
            [0.1, -0.3, 0.11],
            [0.5, 0.3, 0.4]
        )

        # 4. 让 MuJoCo 物理引擎向前推演（由于焊接约束存在，机械臂关节会自动解算 IK 强行跟过来）
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # 在 step() 里面加入判断
        if peg_pos[2] < 0.15 or np.linalg.norm(peg_pos[:2] - hole_pos[:2]) > 0.4:
            terminated = True  # 直接结束，不再给它继续试错的机会

        self.current_step += 1
        observation = self._get_obs()
        
        # 提取关键点用于计算奖励
        ee_pos = self.data.xpos[self.end_effector_id]
        peg_pos = self.data.xpos[self.peg_body_id]
        hole_pos = self.data.xpos[self.hole_body_id]

        # ====================================================================
        # 4. 【核心重构】稠密多阶段奖励设计 (Dense Reward Shaping)
        # ====================================================================
        # 阶段一：机械臂末端 靠近 红插销 的距离
        dist_ee_to_peg = np.linalg.norm(ee_pos - peg_pos)
        # 阶段二：红插销 靠近 蓝色孔洞 的距离
        dist_peg_to_hole = np.linalg.norm(peg_pos - hole_pos)

        # 基础惩罚：离物体越远，扣分越多
        reward_reach = -1.0 * dist_ee_to_peg
        reward_insert = -2.0 * dist_peg_to_hole

        reward = reward_reach + reward_insert

        # 阶段三：终止条件判断（成功标志）
        terminated = False
        # 如果插销进入了孔洞（水平距离很小，且插销的高度沉下去了）
        if dist_peg_to_hole < 0.03 and peg_pos[2] < 0.16:
            reward += 500.0  # 极其丰厚的装配通关大奖！
            terminated = True
            print("🏆 奇迹发生！成功完成高精度轴孔装配！")

        # 步数超时判定
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, truncated, {}

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
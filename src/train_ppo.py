import os
import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 引入你的自定义环境
from src.piper_env import PiperGymEnv

if __name__ == "__main__":
    # 1. 动态计算路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    MODEL_XML_PATH = os.path.join(project_root, "scene_xml", "scene.xml")
    
    print(f"成功加载 MuJoCo 相对路径: {MODEL_XML_PATH}")
    
    # 2. 初始化验证环境（注意：必须在这里显式写好 render_mode="human" 才会弹窗）
    eval_env = DummyVecEnv([lambda: Monitor(PiperGymEnv(xml_path=MODEL_XML_PATH, render_mode="human"))])
    
    # ====================================================================
    # 3. 【核心修改点】注释掉原本的训练代码，彻底不跑训练了！
    # ====================================================================
    # print("🚀 开始 50 万步强化学习闭关训练...")
    # model = PPO("MlpPolicy", eval_env, verbose=1, tensorboard_log="./ppo_piper_tensorboard/")
    # model.learn(total_timesteps=501760)
    # model.save("ppo_piper_real_task")
    # print("💾 训练结束，模型已成功保存！")
    
    # ====================================================================
    # 4. 代替训练：直接从硬盘里把刚才训好的满级模型“捡起来”
    # ====================================================================
    print("⏳ 正在从本地读取已保存的 50 万步大脑权重...")
    model = PPO.load("ppo_piper_real_task", env=eval_env)
    print("✅ 权重加载成功！直接进入最终水平考试...")
    
    # 5. 开始 1500 步的渲染验证循环
    print("--- 验证开始，请观察弹出的 MuJoCo 画面 ---")
    obs = eval_env.reset()
    for i in range(1500):
        # deterministic=False 可以让策略保留灵动性，去对抗你的电机干扰
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = eval_env.step(action)
        
        # 刚才就是因为顶部没写 import time，导致跑到这一行就崩溃了。现在加上了就完美了！
        time.sleep(0.01) 
        
    print("🎉 1500步验证全部结束！")
import os
import gymnasium as gym
import yaml # 如果提示找不到，记得先 pip install pyyaml
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 引入你的自定义环境
from src.piper_env import PiperGymEnv

# 假定你的其他依赖已经导入，如：
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # ==========================================
    # 0. 加载配置文件
    # ==========================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # ==========================================
    # 1. 动态计算路径
    # ==========================================
    MODEL_XML_PATH = os.path.join(project_root, "scene_xml", config["env"]["xml_filename"])
    print(f"成功加载 MuJoCo 相对路径: {MODEL_XML_PATH}")
    
    # ==========================================
    # 2. 初始化验证环境
    # ==========================================
    render_mode = config["env"]["render_mode_eval"]
    eval_env = DummyVecEnv([lambda: Monitor(PiperGymEnv(xml_path=MODEL_XML_PATH, render_mode=render_mode))])
    
    # ==========================================
    # 3. 训练逻辑（已配置化，如需开启取消注释即可）
    # ==========================================
    # print(f"🚀 开始 {config['train']['total_timesteps']} 步强化学习闭关训练...")
    # model = PPO(
    #     config["train"]["policy_type"], 
    #     eval_env, 
    #     verbose=1, 
    #     tensorboard_log=config["train"]["tensorboard_log"]
    # )
    # model.learn(total_timesteps=config["train"]["total_timesteps"])
    # model.save(config["train"]["model_save_name"])
    # print("💾 训练结束，模型已成功保存！")
    
    # ==========================================
    # 4. 代替训练：直接从硬盘里读取已保存的大脑权重
    # ==========================================
    model_load_path = config["eval"]["model_load_name"]
    print(f"⏳ 正在从本地读取已保存的权重: {model_load_path}...")
    model = PPO.load(model_load_path, env=eval_env)
    print("✅ 权重加载成功！直接进入最终水平考试...")
    
    # ==========================================
    # 5. 开始渲染验证循环
    # ==========================================
    eval_steps = config["eval"]["eval_steps"]
    print(f"--- 验证开始，请观察弹出的 MuJoCo 画面（共 {eval_steps} 步） ---")
    
    obs = eval_env.reset()
    for i in range(eval_steps):
        # 从配置中读取 deterministic 参数
        action, _states = model.predict(obs, deterministic=config["eval"]["deterministic"])
        obs, rewards, dones, info = eval_env.step(action)
        
        time.sleep(config["eval"]["sleep_time"]) 
        
    print(f"🎉 {eval_steps}步验证全部结束！")
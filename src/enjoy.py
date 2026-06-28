import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor # 如果报错，请尝试改为 common.monitors

# 1. 导入你的自定义环境
from src.piper_env import PiperGymEnv 

def main():
    # 2. 动态计算项目根目录，完美复刻你 train_ppo.py 里的路径逻辑
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 这是 src 目录
    project_root = os.path.dirname(current_dir)             # 这是项目根目录
    
    # 拼接出准确的 scene.xml 绝对路径
    MODEL_XML_PATH = os.path.join(project_root, "scene_xml", "scene.xml")
    
    print(f"=========================================")
    print(f"成功解析物理场景路径: {MODEL_XML_PATH}")
    print(f"=========================================")
    
    # 3. 复刻向量化环境架构
    env = DummyVecEnv([lambda: Monitor(PiperGymEnv(xml_path=MODEL_XML_PATH, render_mode="human"))])
    
    # 4. 加载你训练好的模型权重
    # 如果你的模型保存在其他地方（如 project_root/logs/best_model），请在这里修改对应的路径
    model_path = os.path.join(project_root, "logs", "best_model") 
    if not os.path.exists(model_path + ".zip"):
        # 备用：如果直接在当前目录下
        model_path = model_path = os.path.join(project_root, "ppo_piper_real_task")
        
    print(f"正在加载强化学习权重: {model_path}")
    model = PPO.load(model_path)
    
    # 5. 开始评估/测试循环
    obs = env.reset()
    print("\n🚀 模型加载成功！MuJoCo 窗口已弹窗。")
    print("💡 提示：调整好视角后，在键盘上按下 [Ctrl + F1] 即可开始/停止录制视频。")
    print("按 Ctrl + C 可以退出测试。\n")
    
    try:
        while True:
            # deterministic=True 确保动作稳健，去除探索噪声
            action, _states = model.predict(obs, deterministic=True)
            
            # VecEnv 在内部会自动处理 reset 信号
            obs, reward, done, info = env.step(action)
            
    except KeyboardInterrupt:
        print("\n测试被用户手动终止。")
    finally:
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    main()
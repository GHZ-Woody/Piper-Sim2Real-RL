import os
import time
import numpy as np
from stable_baselines3 import PPO
from assembly_env import PiperAssemblyEnv

if __name__ == "__main__":
    # 1. 动态获取当前脚本所在目录 (C:\Users\33709\Piper-Sim2Real-RL\src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 【核心修复】往上推一层，拿到真正的项目根目录 (C:\Users\33709\Piper-Sim2Real-RL)
    project_root = os.path.dirname(current_dir)
    
    # 3. 重新精准拼接路径
    xml_path = os.path.join(project_root, "scene_xml", "scene_assembly.xml")
    model_path = os.path.join(project_root, "ppo_piper_assembly_task.zip")
    
    print(f"📄 精准定位 XML 路径: {xml_path}")
    print(f"🎮 精准定位模型路径: {model_path}")
    
    print(f"🎬 正在初始化末端位移验证环境...")
    env = PiperAssemblyEnv(xml_path=xml_path, render_mode="human")
    
    print(f"🧠 正在加载刚刚训好的 3维 Mocap 大脑: {model_path}")
    model = PPO.load(model_path)
    
    # 3. 循环播放机械臂的表现
    for episode in range(5):
        obs, info = env.reset()
        print(f"\n🎬 正在播放第 {episode + 1} 轮表现...")
        
        for step in range(600):
            # 核心修复：确保 action 是由当前 3 维空间自主生成的 [Δx, Δy, Δz]
            # 核心：使用 deterministic=True 让它展示它认为最完美的路线
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)
            
            if terminated or truncated:
                # 顺便打印一下结束时，插销离孔洞的距离
                # 观测维度的第18,19,20维是相对距离向量
                dist = float(info.get("distance", np.linalg.norm(obs[18:21]))) if "distance" in info else np.linalg.norm(obs[18:21])
                print(f"🏁 回合结束，最终插销离孔洞距离: {dist:.4f} 米")
                break
                
    env.close()
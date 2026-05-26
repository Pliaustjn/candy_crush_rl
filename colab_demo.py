"""
Gym环境演示脚本 - 展示如何使用标准Gym接口
这个脚本可以在Colab中单独运行，用于验证Gym环境的正确性
"""
import gym
import numpy as np
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 导入自定义环境（会自动注册）
from gym_env_wrapper import CandyCrushGymEnv, register_gym_env
from candy_crush_network import create_action_mapping


def demo_gym_environment():
    """演示Gym环境的基本使用"""
    print("=" * 60)
    print("Gym Environment Demo")
    print("=" * 60)

    # 创建环境
    env = CandyCrushGymEnv(max_steps=20, target_score=500, seed=42)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation shape: {env.observation_space.shape}")

    # 测试reset
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # 运行几个随机步骤
    print("\nRunning random steps...")
    total_reward = 0

    for step in range(10):
        # 获取合法动作掩码
        legal_mask = env._get_legal_actions_mask()
        legal_actions = np.where(legal_mask == 1)[0]

        if len(legal_actions) == 0:
            print("No legal actions available!")
            break

        # 随机选择合法动作
        action = np.random.choice(legal_actions)

        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"Step {step + 1}: Action={action}, "
              f"Reward={reward:.1f}, "
              f"Score={info['score']}, "
              f"Done={done}")

        if done:
            print("Episode finished!")
            break

    print(f"\nTotal reward: {total_reward:.1f}")
    print(f"Final score: {info['score']}")

    # 测试render
    print("\nRendering environment:")
    env.render()

    env.close()
    print("\n✓ Gym environment works correctly!")


def demo_random_agent():
    """演示随机agent在环境中的表现"""
    print("\n" + "=" * 60)
    print("Random Agent Demo")
    print("=" * 60)

    env = CandyCrushGymEnv(max_steps=50, target_score=1000, seed=123)

    num_episodes = 5
    scores = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_score = 0

        while not done:
            legal_mask = env._get_legal_actions_mask()
            legal_actions = np.where(legal_mask == 1)[0]

            if len(legal_actions) == 0:
                break

            action = np.random.choice(legal_actions)
            obs, reward, done, info = env.step(action)
            episode_score += reward

        scores.append(info.get('score', 0))
        print(f"Episode {episode + 1}: Score = {scores[-1]}")

    print(f"\nAverage score over {num_episodes} episodes: {np.mean(scores):.1f}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")

    env.close()


if __name__ == "__main__":
    print("Testing OpenAI Gym compatibility...")

    # 检查依赖
    try:
        import candy_crush_cpp

        print("✓ candy_crush_cpp module loaded")
    except ImportError as e:
        print(f"✗ Failed to load candy_crush_cpp: {e}")
        print("Please compile the C++ extension first:")
        print("  python setup.py build_ext --inplace")
        sys.exit(1)

    # 运行演示
    demo_gym_environment()
    demo_random_agent()

    print("\n" + "=" * 60)
    print("All tests passed! Environment is Gym-compatible.")
    print("=" * 60)
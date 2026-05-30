"""
PPO训练脚本 - 使用OpenAI Gym标准接口 + TensorBoard可视化
严格PPO实现，不修改原始环境
"""
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import random
from typing import List, Tuple, Dict, Optional
import math
import time
from datetime import datetime, timedelta
import gym

# 导入Gym环境包装器
from gym_env_wrapper import CandyCrushGymEnv, ActionMapper

# 导入神经网络
from candy_crush_network import CandyCrushNet, create_action_mapping

# 导入可视化工具
from visualization import TrainingMonitor


# ==================== 日志系统（保持原有） ====================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_fd = open(log_file, 'w', encoding='utf-8', buffering=1)
        self.log_fd.write(f"PPO Training Log (Gym Standard) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_fd.write("=" * 70 + "\n")
        self.log_fd.flush()

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_fd and not self.log_fd.closed:
            self.log_fd.write(message)
            self.log_fd.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_fd and not self.log_fd.closed:
            self.log_fd.flush()

    def close(self):
        if self.log_fd and not self.log_fd.closed:
            self.log_fd.close()


class LoggerWriter:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        self.logger.write(message)

    def flush(self):
        self.logger.flush()


# ==================== 状态表示 ====================
class StateRepresentation:
    """将Gym观察转换为神经网络输入"""

    def __init__(self, board_size=8, num_colors=6, max_steps=100, target_score=1000):
        self.board_size = board_size
        self.num_colors = num_colors
        self.max_steps = max_steps
        self.target_score = target_score

    def gym_obs_to_tensor(self, gym_obs, steps, score):
        """直接从Gym观察转换为神经网络输入"""
        state = gym_obs.copy()

        # Gym观察已经是(9, 8, 8)格式，但需要更新步数和得分通道
        step_ratio = steps / max(1, self.max_steps)
        state[7] = np.full((self.board_size, self.board_size), step_ratio, dtype=np.float32)

        score_ratio = score / max(1, self.target_score)
        state[8] = np.full((self.board_size, self.board_size), score_ratio, dtype=np.float32)

        return state


# ==================== PPO经验缓冲区 ====================
class PPOExperience:
    """PPO经验存储"""
    __slots__ = ['state', 'action', 'reward', 'done', 'value', 'log_prob', 'legal_mask']

    def __init__(self, state, action, reward, done, value, log_prob, legal_mask):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.value = value
        self.log_prob = log_prob
        self.legal_mask = legal_mask


class PPOBuffer:
    """PPO经验回放缓冲区"""

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def get_all(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ==================== PPO智能体 ====================
class PPOAgent:
    """PPO算法实现"""

    def __init__(self,
                 network: CandyCrushNet,
                 state_repr: StateRepresentation,
                 device: torch.device,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 target_score: int = 1000):

        self.network = network
        self.state_repr = state_repr
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.target_score = target_score
        self.score_normalizer = float(target_score)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0003)

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray) -> Tuple[int, float, float]:
        """选择动作并返回动作、价值和log概率"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        legal_mask_tensor = torch.FloatTensor(legal_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor, legal_actions_mask=legal_mask_tensor)

            # 将非法动作的logits设为负无穷
            policy_logits = policy_logits + torch.log(legal_mask_tensor + 1e-8)

            # 计算概率分布
            policy_probs = F.softmax(policy_logits, dim=-1)
            policy_dist = torch.distributions.Categorical(policy_probs)

            # 采样动作
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action)

        return action.item(), value.item(), log_prob.item()

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool], last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势和回报"""
        values = values + [last_value]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else last_value
            else:
                next_value = 0 if dones[t] else values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, buffer: PPOBuffer, steps: int, score: int) -> Dict[str, float]:
        """PPO更新"""
        experiences = buffer.get_all()

        if len(experiences) < self.mini_batch_size:
            return {}

        # 准备数据
        states = np.stack([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        old_values = np.array([exp.value for exp in experiences])
        old_log_probs = np.array([exp.log_prob for exp in experiences])
        legal_masks = np.stack([exp.legal_mask for exp in experiences])

        # 计算最后状态的价值
        last_exp = experiences[-1]
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(last_exp.state).unsqueeze(0).to(self.device)
            last_mask_tensor = torch.FloatTensor(last_exp.legal_mask).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state_tensor, legal_actions_mask=last_mask_tensor)
            last_value = last_value.item()

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, old_values, dones, last_value)

        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        legal_masks_tensor = torch.FloatTensor(legal_masks).to(self.device)

        # PPO更新循环
        total_samples = len(experiences)
        indices = np.arange(total_samples)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]

                # Mini-batch数据
                mb_states = states_tensor[mini_batch_indices]
                mb_actions = actions_tensor[mini_batch_indices]
                mb_old_log_probs = old_log_probs_tensor[mini_batch_indices]
                mb_advantages = advantages_tensor[mini_batch_indices]
                mb_returns = returns_tensor[mini_batch_indices]
                mb_legal_masks = legal_masks_tensor[mini_batch_indices]

                # 前向传播
                policy_logits, values = self.network(mb_states, legal_actions_mask=mb_legal_masks)

                # 应用合法动作掩码
                policy_logits = policy_logits + torch.log(mb_legal_masks + 1e-8)

                # 计算新的log概率
                policy_probs = F.softmax(policy_logits, dim=-1)
                policy_dist = torch.distributions.Categorical(policy_probs)
                new_log_probs = policy_dist.log_prob(mb_actions)
                entropy = policy_dist.entropy().mean()

                # 计算比率
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO剪辑目标
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # 总损失
                total_loss = (policy_loss +
                              self.value_coef * value_loss -
                              self.entropy_coef * entropy)

                # 优化
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                total_losses.append(total_loss.item())

        # 返回平均损失
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy': np.mean(entropy_losses) if entropy_losses else 0,
            'total_loss': np.mean(total_losses) if total_losses else 0
        }


# ==================== 训练系统（PPO + TensorBoard） ====================
class PPOTrainingSystem:
    """使用PPO算法的训练系统"""

    def __init__(self,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 learning_rate: float = 0.0003,
                 device: Optional[torch.device] = None,
                 verbose: bool = True,
                 log_dir: str = "ppo_logs"):

        self.max_steps = 100
        self.target_score = 1000
        self.score_normalizer = float(self.target_score)
        self.state_repr = StateRepresentation(max_steps=self.max_steps, target_score=self.target_score)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir

        # 神经网络
        self.current_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network.load_state_dict(self.current_network.state_dict())

        # PPO智能体
        self.agent = PPOAgent(
            network=self.current_network,
            state_repr=self.state_repr,
            device=self.device,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            target_score=self.target_score
        )

        # 经验缓冲区
        self.buffer = PPOBuffer(max_size=10000)

        # 统计信息
        self.episode_count = 0
        self.training_step = 0
        self.best_episode_score = 0
        self.training_start_time = None
        self.verbose = verbose

        # TensorBoard监控器（使用独立的日志目录）
        self.monitor = TrainingMonitor(log_dir=os.path.join(self.log_dir, "tensorboard"))
        print(f"TensorBoard monitoring enabled (log_dir: {self.log_dir})")
        print(f"PPO algorithm with Gym environment v{gym.__version__}")

    def collect_experience(self, num_episodes: int = 1) -> Tuple[List[float], List[int], List[float]]:
        """收集经验数据"""
        episode_scores = []
        episode_moves = []
        episode_times = []

        for episode in range(num_episodes):
            env = CandyCrushGymEnv(max_steps=self.max_steps, target_score=self.target_score)
            obs = env.reset()
            done = False
            move_count = 0

            episode_start_time = time.time()

            while not done:
                move_count += 1

                # 获取当前状态
                state = self.state_repr.gym_obs_to_tensor(obs, env.current_steps, env.current_score)
                legal_mask = env._get_legal_actions_mask()

                if np.sum(legal_mask) == 0:
                    break

                # 选择动作
                action, value, log_prob = self.agent.select_action(state, legal_mask)

                # 记录当前状态的经验
                experience = PPOExperience(
                    state=state,
                    action=action,
                    reward=0,  # 将在下一步填充
                    done=False,
                    value=value,
                    log_prob=log_prob,
                    legal_mask=legal_mask
                )

                # 执行动作
                try:
                    obs, reward, done, info = env.step(action)
                    normalized_reward = reward / self.score_normalizer

                    # 更新经验中的奖励和完成标志
                    experience.reward = normalized_reward
                    experience.done = done

                    self.buffer.add(experience)

                except Exception as e:
                    if self.verbose:
                        print(f"Error during action execution: {e}")
                    break

                if env.current_score >= self.target_score:
                    break

            episode_time = time.time() - episode_start_time
            final_score = env.current_score

            episode_scores.append(final_score)
            episode_moves.append(move_count)
            episode_times.append(episode_time)

            if self.verbose and num_episodes <= 5:
                print(f"  Episode score: {final_score}, moves: {move_count}, time: {episode_time:.1f}s")

        return episode_scores, episode_moves, episode_times

    def train_step(self) -> Dict[str, float]:
        """执行PPO训练更新"""
        if len(self.buffer) < self.agent.mini_batch_size:
            return {}

        # 执行PPO更新
        train_info = self.agent.update(
            self.buffer,
            self.training_step,
            self.best_episode_score
        )

        if train_info:
            self.training_step += 1

            # 记录训练指标到TensorBoard
            self.monitor.tensorboard.log_training_metrics(self.training_step, train_info)

        return train_info

    def evaluate_network(self, network: CandyCrushNet, num_games: int = 20) -> float:
        """评估网络性能"""
        scores = []
        print(f"  Evaluating ({num_games} games)...")

        # 创建临时的PPO智能体用于评估
        eval_agent = PPOAgent(
            network=network,
            state_repr=self.state_repr,
            device=self.device,
            clip_epsilon=self.agent.clip_epsilon,
            value_coef=self.agent.value_coef,
            entropy_coef=self.agent.entropy_coef,
            max_grad_norm=self.agent.max_grad_norm,
            ppo_epochs=self.agent.ppo_epochs,
            mini_batch_size=self.agent.mini_batch_size,
            gamma=self.agent.gamma,
            gae_lambda=self.agent.gae_lambda,
            target_score=self.target_score
        )

        for game in range(num_games):
            env = CandyCrushGymEnv(max_steps=self.max_steps, target_score=self.target_score)
            obs = env.reset()
            done = False

            while not done:
                state = self.state_repr.gym_obs_to_tensor(obs, env.current_steps, env.current_score)
                legal_mask = env._get_legal_actions_mask()

                if np.sum(legal_mask) == 0:
                    break

                # 评估时使用贪婪策略
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                legal_mask_tensor = torch.FloatTensor(legal_mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    policy_logits, _ = eval_agent.network(state_tensor, legal_actions_mask=legal_mask_tensor)
                    policy_logits = policy_logits + torch.log(legal_mask_tensor + 1e-8)
                    action = torch.argmax(policy_logits).item()

                try:
                    obs, reward, done, info = env.step(action)
                except Exception:
                    break

                if env.current_score >= self.target_score:
                    break

            scores.append(env.current_score)

            if (game + 1) % 10 == 0:
                print(f"    {game + 1}/{num_games}, avg: {np.mean(scores[-10:]):.0f}")

        avg_score = np.mean(scores)
        max_score = np.max(scores)
        success_rate = sum(1 for s in scores if s >= self.target_score) / num_games * 100

        # 记录评估指标
        self.monitor.tensorboard.log_evaluation_metrics(
            self.episode_count,
            {'avg_score': avg_score, 'max_score': max_score, 'success_rate': success_rate}
        )

        print(f"  Result: avg={avg_score:.0f}, max={max_score:.0f}, success={success_rate:.0f}%")
        return avg_score

    def run_training_loop(self, num_episodes: int = 1000, eval_interval: int = 50,
                          target_update_threshold: float = 0.05,
                          update_interval: int = 5) -> CandyCrushNet:
        """运行PPO训练循环"""
        self.training_start_time = time.time()

        print("=" * 70)
        print(f"PPO Training | Clip: {self.agent.clip_epsilon} | Epochs: {self.agent.ppo_epochs}")
        print(f"Episodes: {num_episodes} | Update Interval: {update_interval}")
        print(f"Device: {self.device} | Mini-batch: {self.agent.mini_batch_size}")
        print(f"TensorBoard: Enabled | Gym version: {gym.__version__}")
        print(f"Log directory: {self.log_dir}")
        print("=" * 70)

        all_scores = []
        best_eval_score = 0

        # 确保模型保存目录存在
        model_dir = os.path.join(self.log_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        episode = 0
        while episode < num_episodes:
            # 收集经验
            collect_episodes = min(update_interval, num_episodes - episode)

            elapsed_total = time.time() - self.training_start_time
            if episode > 0:
                eta = (elapsed_total / episode) * (num_episodes - episode)
            else:
                eta = 0

            print(f"\n{'=' * 70}")
            print(f"Episodes {episode + 1}-{episode + collect_episodes}/{num_episodes} | Buffer: {len(self.buffer)}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed_total))} | ETA: {timedelta(seconds=int(eta))}")

            # 收集经验
            scores, moves, times = self.collect_experience(collect_episodes)
            all_scores.extend(scores)

            episode += collect_episodes
            self.episode_count += collect_episodes

            # 更新最佳分数
            current_max = max(scores)
            if current_max > self.best_episode_score:
                self.best_episode_score = current_max
                print(f"  New record: {current_max}")

            # 打印统计信息
            avg_score = np.mean(scores)
            avg_moves = np.mean(moves)
            avg_time = np.mean(times)
            recent_avg = np.mean(all_scores[-10:]) if all_scores else 0

            print(f"  Avg score: {avg_score:.0f}, Avg moves: {avg_moves:.0f}, Avg time: {avg_time:.1f}s")
            print(f"  10-episode avg: {recent_avg:.0f} | Best: {self.best_episode_score}")

            # PPO训练更新
            if len(self.buffer) >= self.agent.mini_batch_size:
                train_info = self.train_step()

                if train_info:
                    print(f"  Train: policy_loss={train_info['policy_loss']:.4f}, "
                          f"value_loss={train_info['value_loss']:.4f}, "
                          f"entropy={train_info['entropy']:.4f}")

                # 清空缓冲区
                self.buffer.clear()

            # 记录到TensorBoard
            self.monitor.log_training_progress(
                self.episode_count,
                self.training_step,
                {'score': avg_score, 'moves': avg_moves, 'best_score': self.best_episode_score},
                {'buffer_size': len(self.buffer)}
            )

            # 评估
            if episode > 0 and episode % eval_interval == 0:
                print(f"\n{'=' * 50}")
                print(f"Evaluation (Episode {episode})")

                # 评估当前网络
                avg_current = self.evaluate_network(self.current_network, num_games=20)

                # 评估最佳网络
                avg_best = self.evaluate_network(self.best_network, num_games=20)

                print(f"  Current: {avg_current:.0f} | Best: {avg_best:.0f}")

                # 更新最佳网络
                if avg_current > avg_best * (1 + target_update_threshold):
                    print(f"  Updating best network!")
                    self.best_network.load_state_dict(self.current_network.state_dict())
                    best_eval_score = avg_current

                    # 记录最佳网络权重
                    self.monitor.tensorboard.log_network_weights(
                        self.best_network, self.training_step
                    )

                    # 保存模型到独立目录
                    model_path = os.path.join(model_dir, f'best_ppo_network_episode{episode}.pth')
                    torch.save({
                        'model_state_dict': self.best_network.state_dict(),
                        'episode': episode,
                        'avg_score': avg_current,
                        'ppo_params': {
                            'clip_epsilon': self.agent.clip_epsilon,
                            'value_coef': self.agent.value_coef,
                            'entropy_coef': self.agent.entropy_coef,
                            'ppo_epochs': self.agent.ppo_epochs,
                            'gamma': self.agent.gamma,
                            'gae_lambda': self.agent.gae_lambda
                        }
                    }, model_path)
                    print(f"  Saved: {model_path}")
                else:
                    print(f"  Keeping best network (avg: {best_eval_score:.0f})")

        # 保存最终模型
        final_model_path = os.path.join(model_dir, 'best_ppo_network_final.pth')
        torch.save({
            'model_state_dict': self.best_network.state_dict(),
            'ppo_params': {
                'clip_epsilon': self.agent.clip_epsilon,
                'value_coef': self.agent.value_coef,
                'entropy_coef': self.agent.entropy_coef,
                'ppo_epochs': self.agent.ppo_epochs,
                'gamma': self.agent.gamma,
                'gae_lambda': self.agent.gae_lambda
            },
            'total_episodes': self.episode_count,
            'total_training_steps': self.training_step,
            'best_score': self.best_episode_score
        }, final_model_path)

        return self.best_network


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置独立的日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ppo_log_dir = f"ppo_logs_{timestamp}"

    # 创建日志目录
    os.makedirs(ppo_log_dir, exist_ok=True)
    os.makedirs(os.path.join(ppo_log_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(ppo_log_dir, "tensorboard"), exist_ok=True)

    # 设置日志文件
    log_file = os.path.join(ppo_log_dir, f"ppo_training_log.txt")
    logger = Logger(log_file)
    sys.stdout = LoggerWriter(logger)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 测试Gym环境
    print("Testing OpenAI Gym environment...")
    gym_env = CandyCrushGymEnv(max_steps=10, target_score=100)
    obs = gym_env.reset()
    print(f"Gym env OK: observation shape={obs.shape}, action_space={gym_env.action_space}")

    # 测试Gym标准接口
    print(f"Testing Gym standard interface...")
    print(f"  action_space: {gym_env.action_space}")
    print(f"  observation_space: {gym_env.observation_space}")

    # 执行几步随机动作
    for i in range(3):
        legal_mask = gym_env._get_legal_actions_mask()
        legal_actions = np.where(legal_mask == 1)[0]
        if len(legal_actions) > 0:
            action = np.random.choice(legal_actions)
            obs, reward, done, info = gym_env.step(action)
            print(f"  Step {i + 1}: action={action}, reward={reward:.1f}, score={info['score']}")

    print(f"Gym interface test complete!")

    # PPO训练配置
    ppo_config = {
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 10,
        'mini_batch_size': 64,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'learning_rate': 0.0003
    }

    print(f"\nPPO Training Configuration:")
    print(f"  Clip epsilon: {ppo_config['clip_epsilon']}")
    print(f"  Value coefficient: {ppo_config['value_coef']}")
    print(f"  Entropy coefficient: {ppo_config['entropy_coef']}")
    print(f"  PPO epochs: {ppo_config['ppo_epochs']}")
    print(f"  Mini-batch size: {ppo_config['mini_batch_size']}")
    print(f"  Gamma: {ppo_config['gamma']}")
    print(f"  GAE lambda: {ppo_config['gae_lambda']}")
    print(f"  TensorBoard: Enabled")
    print(f"  Log directory: {ppo_log_dir}")

    # 创建训练系统
    training_system = PPOTrainingSystem(
        clip_epsilon=ppo_config['clip_epsilon'],
        value_coef=ppo_config['value_coef'],
        entropy_coef=ppo_config['entropy_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        ppo_epochs=ppo_config['ppo_epochs'],
        mini_batch_size=ppo_config['mini_batch_size'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        learning_rate=ppo_config['learning_rate'],
        verbose=True,
        log_dir=ppo_log_dir
    )

    # 运行训练
    print(f"\n{'#' * 70}")
    print(f"Starting PPO Training")
    print(f"{'#' * 70}")

    best_network = training_system.run_training_loop(
        num_episodes=1000,
        eval_interval=50,
        target_update_threshold=0.05,
        update_interval=5
    )

    # 关闭监控器
    training_system.monitor.close()

    # 恢复标准输出
    sys.stdout = logger.terminal
    logger.close()

    print(f"\nPPO Training completed successfully!")
    print(f"All PPO logs and models saved in: {ppo_log_dir}/")
    print(f"  - Training log: {ppo_log_dir}/ppo_training_log.txt")
    print(f"  - Models: {ppo_log_dir}/models/")
    print(f"  - TensorBoard: {ppo_log_dir}/tensorboard/")
    print(f"To view TensorBoard: tensorboard --logdir {ppo_log_dir}/tensorboard")
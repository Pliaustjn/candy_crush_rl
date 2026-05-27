"""
增强版训练脚本 - 使用OpenAI Gym标准接口 + TensorBoard可视化
不修改原始train_mcts_cpp.py，通过Gym包装器调用环境
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
from collections import defaultdict
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
        self.log_fd = open(log_file, 'w', encoding='utf-8', buffering=1)
        self.log_fd.write(f"Training Log (Gym Standard) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    def board_to_tensor(self, board, steps, score):
        """将棋盘状态转换为神经网络输入张量"""
        state = np.zeros((9, self.board_size, self.board_size), dtype=np.float32)
        board_np = np.array(board, dtype=np.int32)

        # One-hot编码糖果类型
        for i in range(6):
            state[i] = (board_np == i).astype(np.float32)
        # 障碍物通道
        state[6] = (board_np == -1).astype(np.float32)

        # 步数比例通道
        step_ratio = steps / max(1, self.max_steps)
        state[7] = np.full((self.board_size, self.board_size), step_ratio, dtype=np.float32)

        # 得分比例通道
        score_ratio = score / max(1, self.target_score)
        state[8] = np.full((self.board_size, self.board_size), score_ratio, dtype=np.float32)

        return state

    def gym_obs_to_tensor(self, gym_obs, steps, score):
        """直接从Gym观察转换为神经网络输入"""
        state = gym_obs.copy()

        # Gym观察已经是(9, 8, 8)格式，但需要更新步数和得分通道
        step_ratio = steps / max(1, self.max_steps)
        state[7] = np.full((self.board_size, self.board_size), step_ratio, dtype=np.float32)

        score_ratio = score / max(1, self.target_score)
        state[8] = np.full((self.board_size, self.board_size), score_ratio, dtype=np.float32)

        return state


# ==================== MCTS（使用Gym环境） ====================
class MCTSNode:
    __slots__ = ['state', 'steps', 'score', 'parent', 'action', 'reward', 'done',
                 'N', 'W', 'legal_actions_mask', 'children', 'N_a', 'W_a', 'P', 'V']

    def __init__(self, state, steps, score, parent=None, action=None, reward=0, done=False):
        self.state = state  # (9, 8, 8) numpy array
        self.steps = steps
        self.score = score
        self.parent = parent
        self.action = action
        self.reward = reward
        self.done = done
        self.N = 0
        self.W = 0.0
        self.legal_actions_mask = None
        self.children = {}
        self.N_a = defaultdict(int)
        self.W_a = defaultdict(float)
        self.P = None
        self.V = None

    def Q(self, action):
        if self.N_a[action] == 0:
            return 0.0
        return self.W_a[action] / self.N_a[action]

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS:
    """MCTS搜索算法，使用Gym环境"""

    def __init__(self, network, state_repr, env_pool, c_puct=1.0,
                 num_simulations=800, gamma=0.99, score_normalizer=1000.0,
                 max_steps=100, target_score=1000):
        self.network = network
        self.state_repr = state_repr
        self.env_pool = env_pool
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.score_normalizer = score_normalizer
        self.max_steps = max_steps
        self.target_score = target_score
        self.device = next(network.parameters()).device

    def search(self, gym_env):
        """从Gym环境开始搜索"""
        # 获取当前状态
        obs = gym_env._get_observation()
        steps = gym_env.current_steps
        score = gym_env.current_score

        # 创建根节点
        root = MCTSNode(obs, steps, score)

        # 获取合法动作掩码
        root.legal_actions_mask = gym_env._get_legal_actions_mask()
        legal_indices = np.where(root.legal_actions_mask == 1)[0]

        if len(legal_indices) == 0:
            return None

        # 批量评估根节点
        self._batch_evaluate([root])

        # 执行模拟
        for sim in range(self.num_simulations):
            self._simulate(root, gym_env)

        return self._get_action_probs(root, legal_indices)

    def _simulate(self, root, template_env):
        """执行一次MCTS模拟"""
        node = root
        search_path = [node]

        # 创建环境副本用于模拟
        sim_env = CandyCrushGymEnv(
            max_steps=self.max_steps,
            target_score=self.target_score
        )

        # 复制当前状态到模拟环境
        sim_env.current_board = template_env.current_board.copy()
        sim_env.current_steps = template_env.current_steps
        sim_env.current_score = template_env.current_score

        # 选择阶段
        while not node.is_leaf() and not node.done:
            action_idx = self._select_action(node)
            if action_idx is None or action_idx not in node.children:
                break
            node = node.children[action_idx]
            search_path.append(node)

        # 扩展和评估阶段
        if not node.done:
            if node.legal_actions_mask is None:
                # 需要在模拟环境中执行动作以获得新状态
                pass

            if len(node.children) < len(np.where(node.legal_actions_mask == 1)[0]):
                action_idx = self._expand_node(node, sim_env)
                if action_idx is not None and action_idx in node.children:
                    child = node.children[action_idx]
                    self._batch_evaluate([child])
                    search_path.append(child)
                    value = child.V if child.V is not None else 0
                else:
                    value = 0
            else:
                if node.V is None:
                    self._batch_evaluate([node])
                value = node.V if node.V is not None else 0
        else:
            value = node.score / self.score_normalizer

        # 反向传播
        self._backup(search_path, value)

    def _select_action(self, node):
        """选择动作：使用PUCT公式"""
        best_score = -float('inf')
        best_action = None
        sqrt_N = math.sqrt(node.N + 1)
        legal_indices = np.where(node.legal_actions_mask == 1)[0]

        for action_idx in legal_indices:
            Q = node.Q(action_idx)
            P = node.P.get(action_idx, 0) if node.P else 0
            u = self.c_puct * P * sqrt_N / (1 + node.N_a[action_idx])
            score = Q + u
            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action

    def _expand_node(self, node, sim_env):
        """扩展节点：从未访问的动作中随机选择一个"""
        legal_indices = np.where(node.legal_actions_mask == 1)[0]
        unexpanded = [idx for idx in legal_indices if idx not in node.children]

        if not unexpanded:
            return None

        # 随机选择一个未扩展的动作
        action_idx = random.choice(unexpanded)

        # 在模拟环境中执行动作
        try:
            obs, reward, done, info = sim_env.step(action_idx)
            child = MCTSNode(
                obs,
                info['steps'],
                info['score'],
                parent=node,
                action=action_idx,
                reward=reward / self.score_normalizer,
                done=done
            )
            child.legal_actions_mask = info['legal_actions']
            node.children[action_idx] = child
            return action_idx
        except Exception as e:
            return None

    def _batch_evaluate(self, nodes):
        """批量评估节点：使用神经网络预测策略和价值"""
        eval_nodes = []
        for node in nodes:
            if node.done or node.V is not None:
                continue
            if node.legal_actions_mask is None:
                continue
            if np.sum(node.legal_actions_mask) == 0:
                node.V = 0
                continue
            eval_nodes.append(node)

        if not eval_nodes:
            return

        batch_states = np.stack([node.state for node in eval_nodes])
        batch_masks = np.stack([node.legal_actions_mask for node in eval_nodes])

        states_tensor = torch.FloatTensor(batch_states).to(self.device)
        masks_tensor = torch.FloatTensor(batch_masks).to(self.device)

        with torch.no_grad():
            policy_batch, value_batch = self.network(states_tensor, legal_actions_mask=masks_tensor)
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy()

        for i, node in enumerate(eval_nodes):
            node.V = float(value_batch[i][0])
            node.P = {}
            legal_indices = np.where(node.legal_actions_mask == 1)[0]
            for idx in legal_indices:
                node.P[idx] = float(policy_batch[i][idx])

            # 归一化概率
            total_p = sum(node.P.values())
            if total_p > 0:
                for idx in node.P:
                    node.P[idx] /= total_p
            else:
                n = len(node.P)
                for idx in node.P:
                    node.P[idx] = 1.0 / n

    def _backup(self, search_path, leaf_value):
        """反向传播：更新路径上所有节点的统计信息"""
        G = leaf_value
        for node in reversed(search_path):
            node.N += 1
            node.W += G
            if node.parent is not None:
                parent = node.parent
                action = node.action
                parent.N_a[action] += 1
                parent.W_a[action] += G
            G = node.reward + self.gamma * G

    def _get_action_probs(self, root, legal_indices, temperature=1.0):
        """获取动作概率分布"""
        if temperature == 0:
            best_action = max(root.N_a.items(), key=lambda x: x[1])[0]
            probs = {idx: 0.0 for idx in legal_indices}
            probs[best_action] = 1.0
            return probs

        visits = np.array([root.N_a[idx] ** (1.0 / temperature) for idx in legal_indices])
        total_visits = np.sum(visits)

        if total_visits == 0:
            return {idx: root.P.get(idx, 1.0 / len(legal_indices)) for idx in legal_indices}

        visit_probs = visits / total_visits
        return {idx: float(visit_probs[i]) for i, idx in enumerate(legal_indices)}


# ==================== 训练样本和回放缓冲区 ====================
class TrainingSample:
    __slots__ = ['state', 'steps', 'score', 'policy_probs', 'value', 'reward']

    def __init__(self, state, steps, score, policy_probs, value):
        self.state = state
        self.steps = steps
        self.score = score
        self.policy_probs = policy_probs
        self.value = value
        self.reward = 0


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size
        self.position = 0

    def add(self, sample):
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ==================== 训练系统（使用Gym环境 + TensorBoard） ====================
class GymTrainingSystem:
    """使用OpenAI Gym标准接口的训练系统"""

    def __init__(self, c_puct=1.0, num_simulations=200, gamma=0.99,
                 learning_rate=0.001, batch_size=256, replay_buffer_size=100000,
                 device=None, verbose=True):

        self.max_steps = 100
        self.target_score = 1000
        self.score_normalizer = float(self.target_score)
        self.state_repr = StateRepresentation(max_steps=self.max_steps, target_score=self.target_score)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 神经网络
        self.current_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network.load_state_dict(self.current_network.state_dict())

        # 训练参数
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.current_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # 统计信息
        self.episode_count = 0
        self.training_step = 0
        self.best_episode_score = 0
        self.training_start_time = None
        self.verbose = verbose

        # TensorBoard监控器
        self.monitor = TrainingMonitor()
        print(f"TensorBoard monitoring enabled")
        print(f"Gym environment standard: OpenAI Gym v{gym.__version__}")

    def self_play_game(self, temperature=1.0):
        """使用Gym环境进行自我对弈"""
        # 创建Gym环境
        env = CandyCrushGymEnv(
            max_steps=self.max_steps,
            target_score=self.target_score
        )

        obs = env.reset()
        done = False
        samples = []
        move_count = 0

        if self.verbose:
            print(f"  Self-play (temp={temperature}, sim={self.num_simulations})...")

        game_start_time = time.time()

        while not done:
            move_count += 1

            if self.verbose and move_count % 20 == 0:
                print(f"    Move {move_count}: score={env.current_score}")

            # 创建MCTS搜索器
            mcts = MCTS(
                self.current_network,
                self.state_repr,
                None,  # 不使用环境池
                self.c_puct,
                self.num_simulations,
                self.gamma,
                self.score_normalizer,
                max_steps=self.max_steps,
                target_score=self.target_score
            )

            # 搜索最佳动作
            action_probs = mcts.search(env)

            if action_probs is None:
                if self.verbose:
                    print(f"    No valid actions, ending")
                break

            # 根据温度选择动作
            if temperature == 0:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
            else:
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                action = random.choices(actions, weights=probs, k=1)[0]

            # 创建训练样本
            sample = TrainingSample(
                obs.copy(),
                env.current_steps,
                env.current_score,
                action_probs.copy(),
                None
            )

            # 执行动作
            try:
                obs, reward, done, info = env.step(action)
                normalized_reward = reward / self.score_normalizer
                sample.reward = normalized_reward
                samples.append(sample)
            except Exception as e:
                if self.verbose:
                    print(f"    Error: {e}")
                break

            if env.current_score >= self.target_score:
                if self.verbose:
                    print(f"    Target reached!")
                break

        # 计算累积奖励（价值目标）
        G = 0
        for sample in reversed(samples):
            G = sample.reward + self.gamma * G
            sample.value = G

        game_time = time.time() - game_start_time
        final_score = env.current_score

        if self.verbose:
            print(f"  Done: {move_count} moves, score={final_score}, time={game_time:.1f}s")

        # 记录到TensorBoard
        self.monitor.log_training_progress(
            self.episode_count,
            self.training_step,
            {'score': final_score, 'moves': move_count, 'game_time': game_time, 'temperature': temperature},
            {'buffer_size': len(self.replay_buffer)}
        )

        return samples, final_score, move_count, game_time

    def train_step(self):
        """执行一步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = []
        policy_targets = []
        value_targets = []
        legal_masks = []

        for sample in batch:
            # 使用Gym观察格式
            state_tensor = self.state_repr.gym_obs_to_tensor(
                sample.state, sample.steps, sample.score
            )
            states.append(state_tensor)

            # 策略目标
            policy_vec = np.zeros(112, dtype=np.float32)
            for action_idx, prob in sample.policy_probs.items():
                if 0 <= action_idx < 112:
                    policy_vec[action_idx] = prob
            if policy_vec.sum() > 0:
                policy_vec /= policy_vec.sum()
            policy_targets.append(policy_vec)

            # 价值目标
            value_targets.append([sample.value])

            # 获取合法动作掩码
            temp_env = CandyCrushGymEnv(max_steps=self.max_steps, target_score=self.target_score)
            temp_env.current_board = [list(row) for row in sample.state]
            temp_env.current_steps = sample.steps
            temp_env.current_score = sample.score
            legal_mask = temp_env._get_legal_actions_mask()
            legal_masks.append(legal_mask)

        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        policy_targets_tensor = torch.FloatTensor(np.array(policy_targets)).to(self.device)
        value_targets_tensor = torch.FloatTensor(np.array(value_targets)).to(self.device)
        legal_masks_tensor = torch.FloatTensor(np.array(legal_masks)).to(self.device)

        # 前向传播
        self.optimizer.zero_grad()
        policy_probs, values = self.current_network(
            states_tensor,
            legal_actions_mask=legal_masks_tensor
        )

        # 计算损失
        policy_loss = -torch.sum(
            policy_targets_tensor * torch.log(policy_probs + 1e-8)
        ) / len(batch)
        value_loss = F.mse_loss(values, value_targets_tensor)

        # L2正则化
        l2_reg = 0
        for param in self.current_network.parameters():
            l2_reg += torch.norm(param, 2)

        total_loss = policy_loss + value_loss + 0.0001 * l2_reg
        total_loss.backward()
        self.optimizer.step()
        self.training_step += 1

        # 记录训练指标到TensorBoard
        train_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
        self.monitor.tensorboard.log_training_metrics(self.training_step, train_info)

        return train_info

    def evaluate_network(self, network, num_games=20):
        """使用Gym环境评估网络"""
        scores = []
        print(f"  Evaluating ({num_games} games)...")

        for game in range(num_games):
            env = CandyCrushGymEnv(max_steps=self.max_steps, target_score=self.target_score)
            obs = env.reset()
            done = False

            while not done:
                # 使用简化的MCTS进行评估
                mcts = MCTS(
                    network, self.state_repr, None,
                    self.c_puct, min(50, self.num_simulations), self.gamma,
                    self.score_normalizer,
                    max_steps=self.max_steps, target_score=self.target_score
                )

                action_probs = mcts.search(env)
                if action_probs is None:
                    break

                action = max(action_probs.items(), key=lambda x: x[1])[0]

                try:
                    obs, reward, done, info = env.step(action)
                except Exception:
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

    def run_training_loop(self, num_episodes=1000, eval_interval=50, target_update_threshold=0.05):
        """运行训练循环"""
        self.training_start_time = time.time()

        print("=" * 70)
        print(f"Gym Standard Training | MCTS sims: {self.num_simulations} | Episodes: {num_episodes}")
        print(f"Device: {self.device} | Batch: {self.batch_size} | Buffer: {self.replay_buffer.max_size}")
        print(f"TensorBoard: Enabled | Gym version: {gym.__version__}")
        print("=" * 70)

        episode_scores = []
        best_eval_score = 0

        for episode in range(num_episodes):
            # 温度退火
            if episode < num_episodes * 0.3:
                temperature = 1.0
            elif episode < num_episodes * 0.7:
                temperature = 0.5
            else:
                temperature = 0.1

            elapsed_total = time.time() - self.training_start_time
            eta = (elapsed_total / max(1, episode + 1)) * (num_episodes - episode - 1)

            print(f"\n{'=' * 70}")
            print(f"Episode {episode + 1}/{num_episodes} | Temp: {temperature} | Buffer: {len(self.replay_buffer)}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed_total))} | ETA: {timedelta(seconds=int(eta))}")

            # 自我对弈
            samples, score, moves, game_time = self.self_play_game(temperature)
            episode_scores.append(score)

            if score > self.best_episode_score:
                self.best_episode_score = score
                print(f"  New record: {score}")

            # 添加到回放缓冲区
            for sample in samples:
                self.replay_buffer.add(sample)

            self.episode_count += 1

            # 训练
            if len(self.replay_buffer) >= self.batch_size:
                num_updates = min(10, len(samples))
                total_policy_loss = 0
                total_value_loss = 0
                for _ in range(num_updates):
                    train_info = self.train_step()
                    if train_info:
                        total_policy_loss += train_info['policy_loss']
                        total_value_loss += train_info['value_loss']

                avg_policy_loss = total_policy_loss / max(1, num_updates)
                avg_value_loss = total_value_loss / max(1, num_updates)
                print(f"  Train: policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}")

            # 统计信息
            recent_avg = np.mean(episode_scores[-10:]) if episode_scores else 0
            print(f"  Score: {score}/{moves} moves | 10-avg: {recent_avg:.0f} | Best: {self.best_episode_score}")

            # 评估
            if episode > 0 and (episode + 1) % eval_interval == 0:
                print(f"\n{'=' * 50}")
                print(f"Evaluation (Episode {episode + 1})")
                avg_current = self.evaluate_network(self.current_network, num_games=20)
                avg_best = self.evaluate_network(self.best_network, num_games=20)
                print(f"  Current: {avg_current:.0f} | Best: {avg_best:.0f}")

                if avg_current > avg_best * (1 + target_update_threshold):
                    print(f"  Updating best network!")
                    self.best_network.load_state_dict(self.current_network.state_dict())
                    best_eval_score = avg_current

                    # 记录最佳网络权重
                    self.monitor.tensorboard.log_network_weights(
                        self.best_network, self.training_step
                    )

                    model_path = f'best_network_sim{self.num_simulations}.pth'
                    torch.save({
                        'model_state_dict': self.best_network.state_dict(),
                        'num_simulations': self.num_simulations,
                        'episode': episode + 1,
                        'avg_score': avg_current,
                    }, model_path)
                    print(f"  Saved: {model_path}")
                else:
                    print(f"  Keeping best network (avg: {best_eval_score:.0f})")

        return self.best_network


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置日志
    log_file = f"training_log_gym_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = LoggerWriter(logger)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

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

    # 训练阶段配置
    training_stages = [
        {'num_simulations': 10, 'num_episodes': 100},
        {'num_simulations': 25, 'num_episodes': 100},
        {'num_simulations': 50, 'num_episodes': 100},
        {'num_simulations': 100, 'num_episodes': 100},
        {'num_simulations': 200, 'num_episodes': 100},
    ]

    print(f"\nProgressive training with OpenAI Gym standard:")
    print(f"  Stages: {len(training_stages)}")
    print(f"  Total episodes: {sum(s['num_episodes'] for s in training_stages)}")
    print(f"  TensorBoard: Enabled")
    print(f"  Log file: {log_file}")

    best_network_state = None

    for stage_idx, stage in enumerate(training_stages):
        print(f"\n{'#' * 70}")
        print(f"# Stage {stage_idx + 1}/{len(training_stages)}: "
              f"sims={stage['num_simulations']}, "
              f"episodes={stage['num_episodes']}")
        print(f"{'#' * 70}")

        # 创建训练系统
        training_system = GymTrainingSystem(
            c_puct=1.0,
            num_simulations=stage['num_simulations'],
            gamma=0.99,
            learning_rate=0.001,
            batch_size=128,
            replay_buffer_size=50000,
            verbose=True
        )

        # 加载之前的最佳网络
        if best_network_state is not None:
            training_system.current_network.load_state_dict(best_network_state)
            training_system.best_network.load_state_dict(best_network_state)
            print(f"Loaded stage {stage_idx} best network")

        # 运行训练
        best_network = training_system.run_training_loop(
            num_episodes=stage['num_episodes'],
            eval_interval=max(10, stage['num_episodes'] // 5),
            target_update_threshold=0.05
        )

        # 保存网络状态
        best_network_state = {
            k: v.cpu().clone()
            for k, v in best_network.state_dict().items()
        }

        # 保存阶段模型
        stage_model_path = f'best_network_stage{stage_idx + 1}_sim{stage["num_simulations"]}.pth'
        torch.save({
            'model_state_dict': best_network_state,
            'stage': stage_idx + 1,
            'num_simulations': stage['num_simulations'],
            'gym_standard': True
        }, stage_model_path)
        print(f"\nStage {stage_idx + 1} complete! Model: {stage_model_path}")

    # 保存最终模型
    print(f"\n{'#' * 70}")
    print(f"Training complete with OpenAI Gym standard!")
    torch.save({
        'model_state_dict': best_network_state,
        'stages': len(training_stages),
        'gym_standard': True
    }, 'best_network_final.pth')
    print(f"Final model: best_network_final.pth")

    # 关闭监控器
    training_system.monitor.close()

    # 恢复标准输出
    sys.stdout = logger.terminal
    logger.close()

    print(f"\nTraining completed successfully!")
    print(f"TensorBoard logs saved in: training_logs_*")
    print(f"To view TensorBoard: tensorboard --logdir training_logs_*/tensorboard")
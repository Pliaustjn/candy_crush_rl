"""
MCTS训练脚本 - 使用C++环境（clone()版本）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
import copy
from typing import List, Tuple, Dict, Optional
import math
import time
import sys
import os
from datetime import datetime, timedelta

import candy_crush_cpp
from candy_crush_network import CandyCrushNet, create_action_mapping


# ==================== 日志系统 ====================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log_fd = open(log_file, 'w', encoding='utf-8', buffering=1)
        self.log_fd.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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


# ==================== 动作映射 ====================
class ActionMapper:
    def __init__(self):
        self.action_map = create_action_mapping()
        self.action_to_idx = {}
        for idx, action in enumerate(self.action_map):
            r, c, d = action['row'], action['col'], action['direction']
            if d == 'h':
                coord = (r, c, r, c + 1)
            else:
                coord = (r, c, r + 1, c)
            self.action_to_idx[coord] = idx

    def coord_to_idx(self, r1, c1, r2, c2):
        return self.action_to_idx.get((r1, c1, r2, c2), -1)

    def idx_to_coord(self, idx):
        action = self.action_map[idx]
        r, c, d = action['row'], action['col'], action['direction']
        if d == 'h':
            return (r, c, r, c + 1)
        else:
            return (r, c, r + 1, c)


# ==================== 状态表示 ====================
class StateRepresentation:
    def __init__(self, board_size=8, num_colors=6, max_steps=100, target_score=1000):
        self.board_size = board_size
        self.num_colors = num_colors
        self.max_steps = max_steps
        self.target_score = target_score

    def board_to_tensor(self, board, steps, score):
        state = np.zeros((9, self.board_size, self.board_size), dtype=np.float32)
        board_np = np.array(board, dtype=np.int32)

        for i in range(6):
            state[i] = (board_np == i).astype(np.float32)
        state[6] = (board_np == -1).astype(np.float32)

        step_ratio = steps / max(1, self.max_steps)
        state[7] = np.full((self.board_size, self.board_size), step_ratio, dtype=np.float32)

        score_ratio = score / max(1, self.target_score)
        state[8] = np.full((self.board_size, self.board_size), score_ratio, dtype=np.float32)

        return state


# ==================== 环境输出控制 ====================
class SuppressPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress
        self.old_stdout = None
        self.null_fd = None

    def __enter__(self):
        if self.suppress:
            self.old_stdout = sys.stdout
            self.null_fd = open(os.devnull, 'w', encoding='utf-8')
            sys.stdout = self.null_fd
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            if self.null_fd:
                self.null_fd.close()
            if self.old_stdout:
                sys.stdout = self.old_stdout
        return False


# ==================== 环境池 ====================
class EnvPool:
    def __init__(self, max_steps=100, target_score=1000, pool_size=8):
        self.max_steps = max_steps
        self.target_score = target_score
        self.pool = []
        self.pool_size = pool_size

    def get(self):
        if self.pool:
            return self.pool.pop()
        else:
            return candy_crush_cpp.CandyCrushEnv(
                max_steps=self.max_steps,
                target_score=self.target_score)

    def put(self, env):
        if len(self.pool) < self.pool_size:
            self.pool.append(env)


# ==================== MCTS ====================
class MCTSNode:
    __slots__ = ['board', 'steps', 'score', 'parent', 'action', 'reward', 'done',
                 'N', 'W', 'legal_actions', 'children', 'N_a', 'W_a', 'P', 'V',
                 'env_clone']

    def __init__(self, board, steps, score, parent=None, action=None, reward=0, done=False, env_clone=None):
        self.board = board
        self.steps = steps
        self.score = score
        self.parent = parent
        self.action = action
        self.reward = reward
        self.done = done
        self.N = 0
        self.W = 0.0
        self.legal_actions = None
        self.children = {}
        self.N_a = defaultdict(int)
        self.W_a = defaultdict(float)
        self.P = None
        self.V = None
        self.env_clone = env_clone

    def Q(self, action):
        if self.N_a[action] == 0:
            return 0.0
        return self.W_a[action] / self.N_a[action]

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, network, action_mapper, state_repr,
                 c_puct=1.0, num_simulations=800, gamma=0.99,
                 score_normalizer=1000.0, env_pool=None,
                 suppress_env_output=True, max_steps=100, target_score=1000):
        self.network = network
        self.action_mapper = action_mapper
        self.state_repr = state_repr
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.score_normalizer = score_normalizer
        self.env_pool = env_pool
        self.suppress_env_output = suppress_env_output
        self.max_steps = max_steps
        self.target_score = target_score
        self.device = next(network.parameters()).device

    def search(self, board, steps, score):
        board_copy = [row.copy() for row in board]

        temp_env = self._get_temp_env()
        temp_env.set_state(board_copy, steps, score)
        root_env_clone = temp_env.clone()
        self._return_temp_env(temp_env)

        root = MCTSNode(board_copy, steps, score, env_clone=root_env_clone)
        root.legal_actions = root_env_clone.get_valid_actions()

        if not root.legal_actions:
            return None

        self._batch_evaluate([root])

        for sim in range(self.num_simulations):
            self._simulate(root)

        return self._get_action_probs(root)

    def _get_temp_env(self):
        if self.env_pool:
            return self.env_pool.get()
        return candy_crush_cpp.CandyCrushEnv(max_steps=self.max_steps, target_score=self.target_score)

    def _return_temp_env(self, env):
        if self.env_pool:
            self.env_pool.put(env)

    def _simulate(self, root):
        node = root
        search_path = [node]

        while not node.is_leaf() and not node.done:
            action = self._select_action(node)
            if action not in node.children:
                break
            node = node.children[action]
            search_path.append(node)

        if not node.done:
            if node.legal_actions is None:
                node.legal_actions = node.env_clone.get_valid_actions()

            if node.legal_actions and len(node.children) < len(node.legal_actions):
                action = self._expand_node(node)
                if action is not None and action in node.children:
                    child = node.children[action]
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

        self._backup(search_path, value)

    def _select_action(self, node):
        best_score = -float('inf')
        best_action = None
        sqrt_N = math.sqrt(node.N + 1)

        for action in node.legal_actions:
            Q = node.Q(action)
            P = node.P.get(action, 0) if node.P else 0
            u = self.c_puct * P * sqrt_N / (1 + node.N_a[action])
            score = Q + u
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _expand_node(self, node):
        unexpanded = [a for a in node.legal_actions if a not in node.children]
        if not unexpanded:
            return None

        action = random.choice(unexpanded)
        sim_env = node.env_clone

        with SuppressPrints(self.suppress_env_output):
            try:
                next_board, reward, done = sim_env.step(action)
                next_steps = sim_env.get_steps()
                next_score = sim_env.get_score()
                child_env_clone = sim_env.clone()
            except Exception:
                return None

        normalized_reward = reward / self.score_normalizer
        next_board_list = [list(row) for row in next_board]

        child = MCTSNode(next_board_list, next_steps, next_score,
                         parent=node, action=action,
                         reward=normalized_reward, done=done,
                         env_clone=child_env_clone)
        node.children[action] = child
        return action

    def _batch_evaluate(self, nodes):
        eval_nodes = []
        for node in nodes:
            if node.done or node.V is not None:
                continue
            if node.legal_actions is None:
                node.legal_actions = node.env_clone.get_valid_actions()
            if not node.legal_actions:
                node.V = 0
                continue
            eval_nodes.append(node)

        if not eval_nodes:
            return

        batch_states = []
        batch_masks = []

        for node in eval_nodes:
            state_tensor = self.state_repr.board_to_tensor(node.board, node.steps, node.score)
            batch_states.append(state_tensor)

            legal_mask = np.zeros(112, dtype=np.float32)
            for action in node.legal_actions:
                idx = self.action_mapper.coord_to_idx(*action)
                if 0 <= idx < 112:
                    legal_mask[idx] = 1.0
            batch_masks.append(legal_mask)

        states_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
        masks_tensor = torch.FloatTensor(np.array(batch_masks)).to(self.device)

        with torch.no_grad():
            policy_batch, value_batch = self.network(states_tensor, legal_actions_mask=masks_tensor)
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy()

        for i, node in enumerate(eval_nodes):
            node.V = float(value_batch[i][0])
            node.P = {}
            for action in node.legal_actions:
                idx = self.action_mapper.coord_to_idx(*action)
                if 0 <= idx < 112:
                    node.P[action] = float(policy_batch[i][idx])
                else:
                    node.P[action] = 0.0

            total_p = sum(node.P.values())
            if total_p > 0:
                for action in node.P:
                    node.P[action] /= total_p
            else:
                n = len(node.P)
                for action in node.P:
                    node.P[action] = 1.0 / n

    def _backup(self, search_path, leaf_value):
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

    def _get_action_probs(self, root, temperature=1.0):
        if temperature == 0:
            best_action = max(root.N_a.items(), key=lambda x: x[1])[0]
            probs = {a: 0.0 for a in root.legal_actions}
            probs[best_action] = 1.0
            return probs

        visits = np.array([root.N_a[a] ** (1.0 / temperature) for a in root.legal_actions])
        total_visits = np.sum(visits)

        if total_visits == 0:
            return {a: root.P.get(a, 1.0 / len(root.legal_actions)) for a in root.legal_actions}

        visit_probs = visits / total_visits
        return {a: float(visit_probs[i]) for i, a in enumerate(root.legal_actions)}


# ==================== 训练系统 ====================
class TrainingSample:
    __slots__ = ['board', 'steps', 'score', 'policy_probs', 'value', 'reward']

    def __init__(self, board, steps, score, policy_probs, value):
        self.board = board
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


class TrainingSystem:
    def __init__(self, c_puct=1.0, num_simulations=200, gamma=0.99,
                 learning_rate=0.001, batch_size=256, replay_buffer_size=100000,
                 device=None, suppress_env_output=True, env_pool_size=8, verbose=True):

        self.max_steps = 100
        self.target_score = 1000
        self.score_normalizer = float(self.target_score)
        self.action_mapper = ActionMapper()
        self.state_repr = StateRepresentation(max_steps=self.max_steps, target_score=self.target_score)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_pool = EnvPool(max_steps=self.max_steps, target_score=self.target_score, pool_size=env_pool_size)
        self.suppress_env_output = suppress_env_output

        self.current_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network = CandyCrushNet(num_filters=64).to(self.device)
        self.best_network.load_state_dict(self.current_network.state_dict())

        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.current_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.episode_count = 0
        self.training_step = 0
        self.best_episode_score = 0
        self.training_start_time = None
        self.verbose = verbose

    def self_play_game(self, temperature=1.0):
        env = self.env_pool.get()
        board = env.reset()
        steps = env.get_steps()
        score = env.get_score()
        done = False
        samples = []
        move_count = 0

        if self.verbose:
            print(f"  Self-play (temp={temperature}, sim={self.num_simulations})...")

        game_start_time = time.time()

        while not done:
            move_count += 1

            if self.verbose and move_count % 20 == 0:
                print(f"    Move {move_count}: score={score}")

            board_list = [list(row) for row in board]

            mcts = MCTS(
                self.current_network, self.action_mapper, self.state_repr,
                self.c_puct, self.num_simulations, self.gamma,
                self.score_normalizer, self.env_pool,
                self.suppress_env_output,
                max_steps=self.max_steps, target_score=self.target_score)

            action_probs = mcts.search(board_list, steps, score)

            if action_probs is None:
                if self.verbose:
                    print(f"    No valid actions, ending")
                break

            if temperature == 0:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
            else:
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                action = random.choices(actions, weights=probs, k=1)[0]

            sample = TrainingSample(board_list, steps, score, action_probs.copy(), None)

            with SuppressPrints(self.suppress_env_output):
                try:
                    next_board, reward, done = env.step(action)
                    normalized_reward = reward / self.score_normalizer
                    sample.reward = normalized_reward
                    samples.append(sample)
                    board = next_board
                    steps = env.get_steps()
                    score = env.get_score()
                except Exception as e:
                    if self.verbose:
                        print(f"    Error: {e}")
                    break

            if score >= self.target_score:
                if self.verbose:
                    print(f"    Target reached!")
                break

        G = 0
        for sample in reversed(samples):
            G = sample.reward + self.gamma * G
            sample.value = G

        game_time = time.time() - game_start_time

        if self.verbose:
            print(f"  Done: {move_count} moves, score={score}, time={game_time:.1f}s")

        self.env_pool.put(env)
        return samples, score, move_count, game_time

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = []
        policy_targets = []
        value_targets = []
        legal_masks = []

        for sample in batch:
            state_tensor = self.state_repr.board_to_tensor(sample.board, sample.steps, sample.score)
            states.append(state_tensor)

            policy_vec = np.zeros(112, dtype=np.float32)
            for action, prob in sample.policy_probs.items():
                idx = self.action_mapper.coord_to_idx(*action)
                if 0 <= idx < 112:
                    policy_vec[idx] = prob
            if policy_vec.sum() > 0:
                policy_vec /= policy_vec.sum()
            policy_targets.append(policy_vec)
            value_targets.append([sample.value])

            temp_env = self.env_pool.get()
            temp_env.set_state(sample.board, sample.steps, sample.score)
            legal_actions = temp_env.get_valid_actions()
            self.env_pool.put(temp_env)

            legal_mask = np.zeros(112, dtype=np.float32)
            for action in legal_actions:
                idx = self.action_mapper.coord_to_idx(*action)
                if 0 <= idx < 112:
                    legal_mask[idx] = 1.0
            legal_masks.append(legal_mask)

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        policy_targets_tensor = torch.FloatTensor(np.array(policy_targets)).to(self.device)
        value_targets_tensor = torch.FloatTensor(np.array(value_targets)).to(self.device)
        legal_masks_tensor = torch.FloatTensor(np.array(legal_masks)).to(self.device)

        self.optimizer.zero_grad()
        policy_probs, values = self.current_network(states_tensor, legal_actions_mask=legal_masks_tensor)

        policy_loss = -torch.sum(policy_targets_tensor * torch.log(policy_probs + 1e-8)) / len(batch)
        value_loss = F.mse_loss(values, value_targets_tensor)

        l2_reg = 0
        for param in self.current_network.parameters():
            l2_reg += torch.norm(param, 2)

        total_loss = policy_loss + value_loss + 0.0001 * l2_reg
        total_loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}

    def evaluate_network(self, network, num_games=20):
        scores = []
        print(f"  Evaluating ({num_games} games)...")

        for game in range(num_games):
            env = self.env_pool.get()
            board = env.reset()
            steps = env.get_steps()
            score = env.get_score()
            done = False

            while not done:
                board_list = [list(row) for row in board]

                mcts = MCTS(
                    network, self.action_mapper, self.state_repr,
                    self.c_puct, min(50, self.num_simulations), self.gamma,
                    self.score_normalizer, self.env_pool,
                    True, max_steps=self.max_steps, target_score=self.target_score)

                action_probs = mcts.search(board_list, steps, score)
                if action_probs is None:
                    break

                action = max(action_probs.items(), key=lambda x: x[1])[0]

                with SuppressPrints(True):
                    try:
                        next_board, reward, done = env.step(action)
                        board = next_board
                        steps = env.get_steps()
                        score = env.get_score()
                    except Exception:
                        break

            scores.append(score)
            self.env_pool.put(env)

            if (game + 1) % 10 == 0:
                print(f"    {game + 1}/{num_games}, avg: {np.mean(scores[-10:]):.0f}")

        avg_score = np.mean(scores)
        max_score = np.max(scores)
        success_rate = sum(1 for s in scores if s >= self.target_score) / num_games * 100
        print(f"  Result: avg={avg_score:.0f}, max={max_score:.0f}, success={success_rate:.0f}%")
        return avg_score

    def run_training_loop(self, num_episodes=1000, eval_interval=50, target_update_threshold=0.05):
        self.training_start_time = time.time()

        print("=" * 70)
        print(f"MCTS sims: {self.num_simulations} | Episodes: {num_episodes}")
        print(f"Device: {self.device} | Batch: {self.batch_size} | Buffer: {self.replay_buffer.max_size}")
        print("=" * 70)

        episode_scores = []
        best_eval_score = 0

        for episode in range(num_episodes):
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

            samples, score, moves, game_time = self.self_play_game(temperature)
            episode_scores.append(score)

            if score > self.best_episode_score:
                self.best_episode_score = score
                print(f"  New record: {score}")

            for sample in samples:
                self.replay_buffer.add(sample)

            self.episode_count += 1

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

            recent_avg = np.mean(episode_scores[-10:]) if episode_scores else 0
            print(f"  Score: {score}/{moves} moves | 10-avg: {recent_avg:.0f} | Best: {self.best_episode_score}")

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
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = LoggerWriter(logger)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 测试C++环境
    print("Testing C++ environment...")
    test_env = candy_crush_cpp.CandyCrushEnv(max_steps=10, target_score=100)
    test_board = test_env.reset()
    print(f"C++ env OK: steps={test_env.get_steps()}, score={test_env.get_score()}")

    # 测试 set_state 和 clone
    test_env.set_state(test_board, 5, 100)
    print(f"set_state OK: steps={test_env.get_steps()}, score={test_env.get_score()}")

    cloned_env = test_env.clone()
    print(f"clone OK: steps={cloned_env.get_steps()}, score={cloned_env.get_score()}")

    training_stages = [
        {'num_simulations': 10, 'num_episodes': 100},
        {'num_simulations': 25, 'num_episodes': 100},
        {'num_simulations': 50, 'num_episodes': 100},
        {'num_simulations': 100, 'num_episodes': 100},
        {'num_simulations': 200, 'num_episodes': 100},
    ]

    print(f"\nProgressive training: {len(training_stages)} stages, {sum(s['num_episodes'] for s in training_stages)} episodes")
    print(f"Log: {log_file}")

    SUPPRESS_ENV_OUTPUT = True
    best_network_state = None

    for stage_idx, stage in enumerate(training_stages):
        print(f"\n{'#' * 70}")
        print(f"# Stage {stage_idx + 1}/{len(training_stages)}: sims={stage['num_simulations']}, episodes={stage['num_episodes']}")
        print(f"{'#' * 70}")

        training_system = TrainingSystem(
            c_puct=1.0,
            num_simulations=stage['num_simulations'],
            gamma=0.99,
            learning_rate=0.001,
            batch_size=128,
            replay_buffer_size=50000,
            suppress_env_output=SUPPRESS_ENV_OUTPUT,
            env_pool_size=8,
            verbose=True)

        if best_network_state is not None:
            training_system.current_network.load_state_dict(best_network_state)
            training_system.best_network.load_state_dict(best_network_state)
            print(f"Loaded stage {stage_idx} best network")

        best_network = training_system.run_training_loop(
            num_episodes=stage['num_episodes'],
            eval_interval=max(10, stage['num_episodes'] // 5),
            target_update_threshold=0.05)

        best_network_state = {k: v.cpu().clone() for k, v in best_network.state_dict().items()}

        stage_model_path = f'best_network_stage{stage_idx + 1}_sim{stage["num_simulations"]}.pth'
        torch.save({'model_state_dict': best_network_state, 'stage': stage_idx + 1,
                     'num_simulations': stage['num_simulations']}, stage_model_path)
        print(f"\nStage {stage_idx + 1} complete! Model: {stage_model_path}")

    print(f"\n{'#' * 70}")
    print(f"Training complete!")
    torch.save({'model_state_dict': best_network_state, 'stages': len(training_stages)}, 'best_network_final.pth')
    print(f"Final model: best_network_final.pth")

    sys.stdout = logger.terminal
    logger.close()
    print(f"\nDone! Log: {log_file}")
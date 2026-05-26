"""
OpenAI Gym兼容的环境包装器
不修改原始C++代码，仅提供标准Gym接口
"""
import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os

# 修复导入路径：添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import candy_crush_cpp
from candy_crush_network import create_action_mapping


class CandyCrushGymEnv(gym.Env):
    """
    Candy Crush Saga的OpenAI Gym兼容环境

    遵循标准Gym接口：
    - reset() -> observation
    - step(action) -> (observation, reward, done, info)
    - render() -> None
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_steps: int = 100, target_score: int = 1000, seed: Optional[int] = None):
        super().__init__()

        # 初始化C++环境
        self.env = candy_crush_cpp.CandyCrushEnv(
            max_steps=max_steps,
            target_score=target_score,
            seed=seed if seed is not None else -1
        )

        # 定义动作空间：112个离散动作
        self.action_space = spaces.Discrete(112)

        # 定义观察空间：9个8x8的通道
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, 8, 8),
            dtype=np.float32
        )

        # 动作映射器
        self.action_mapper = ActionMapper()

        # 状态
        self.current_board = None
        self.current_steps = 0
        self.current_score = 0
        self.max_steps = max_steps
        self.target_score = target_score

    def reset(self) -> np.ndarray:
        """重置环境并返回初始观察"""
        board = self.env.reset()
        self.current_board = [list(row) for row in board]
        self.current_steps = self.env.get_steps()
        self.current_score = self.env.get_score()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 0-111的离散动作索引

        Returns:
            observation: 状态观察
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        # 将离散动作转换为坐标元组
        coord_action = self.action_mapper.idx_to_coord(action)

        try:
            next_board, reward, done = self.env.step(coord_action)
            self.current_board = [list(row) for row in next_board]
            self.current_steps = self.env.get_steps()
            self.current_score = self.env.get_score()
        except Exception as e:
            # 非法动作处理
            reward = -1.0  # 惩罚非法动作
            done = True
            info = {'error': str(e), 'valid': False}
            return self._get_observation(), reward, done, info

        info = {
            'score': self.current_score,
            'steps': self.current_steps,
            'valid': True,
            'legal_actions': self._get_legal_actions_mask()
        }

        return self._get_observation(), float(reward), done, info

    def render(self, mode: str = 'human'):
        """渲染环境"""
        if mode == 'human':
            self.env.render()
        elif mode == 'rgb_array':
            # 返回RGB数组（简化实现）
            board = self.env.get_board()
            rgb_array = np.zeros((8, 8, 3), dtype=np.uint8)
            color_map = {
                0: [255, 0, 0],  # 红色
                1: [0, 0, 255],  # 蓝色
                2: [0, 255, 0],  # 绿色
                3: [128, 0, 128],  # 紫色
                4: [255, 165, 0],  # 橙色
                5: [255, 255, 0],  # 黄色
                -1: [100, 100, 100]  # 障碍物
            }
            for i in range(8):
                for j in range(8):
                    color = board[i][j]
                    rgb_array[i, j] = color_map.get(color, [0, 0, 0])
            return rgb_array

    def seed(self, seed: Optional[int] = None) -> list:
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def _get_observation(self) -> np.ndarray:
        """将环境状态转换为神经网络输入格式"""
        board_np = np.array(self.current_board, dtype=np.int32)
        state = np.zeros((9, 8, 8), dtype=np.float32)

        # One-hot编码棋盘状态（7个通道）
        for i in range(6):
            state[i] = (board_np == i).astype(np.float32)
        state[6] = (board_np == -1).astype(np.float32)

        # 步数比例通道
        step_ratio = self.current_steps / max(1, self.max_steps)
        state[7] = np.full((8, 8), step_ratio, dtype=np.float32)

        # 得分比例通道
        score_ratio = self.current_score / max(1, self.target_score)
        state[8] = np.full((8, 8), score_ratio, dtype=np.float32)

        return state

    def _get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码"""
        legal_mask = np.zeros(112, dtype=np.float32)
        valid_actions = self.env.get_valid_actions()
        for action in valid_actions:
            idx = self.action_mapper.coord_to_idx(*action)
            if 0 <= idx < 112:
                legal_mask[idx] = 1.0
        return legal_mask


class ActionMapper:
    """动作映射器：在离散索引和坐标元组之间转换"""

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

    def coord_to_idx(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """坐标转索引"""
        return self.action_to_idx.get((r1, c1, r2, c2), -1)

    def idx_to_coord(self, idx: int) -> Tuple[int, int, int, int]:
        """索引转坐标"""
        action = self.action_map[idx]
        r, c, d = action['row'], action['col'], action['direction']
        if d == 'h':
            return (r, c, r, c + 1)
        else:
            return (r, c, r + 1, c)


# Gym环境注册（可选）
def register_gym_env():
    """注册自定义Gym环境"""
    gym.envs.registration.register(
        id='CandyCrush-v0',
        entry_point='gym_env_wrapper:CandyCrushGymEnv',
        max_episode_steps=100,
    )
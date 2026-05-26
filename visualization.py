"""
TensorBoard可视化工具
不修改原始训练代码，仅提供独立的可视化功能
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional
import os
import sys
from datetime import datetime

# 添加src目录到Python路径，以便导入原始模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class TrainingVisualizer:
    """
    TensorBoard训练可视化器

    记录以下指标：
    - 训练奖励
    - 策略损失和价值损失
    - 评估分数
    - 学习率
    - 网络参数直方图
    """

    def __init__(self, log_dir: str = None):
        """
        初始化TensorBoard写入器

        Args:
            log_dir: 日志目录，默认使用时间戳命名
        """
        if log_dir is None:
            log_dir = f"runs/candy_crush_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard logs: {log_dir}")
        print(f"To view: tensorboard --logdir {log_dir}")

    def log_episode_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        记录单episode的训练指标

        Args:
            episode: 当前episode编号
            metrics: 指标字典，可包含：
                - 'score': 游戏得分
                - 'moves': 移动次数
                - 'game_time': 游戏时间
                - 'reward': 总奖励
                - 'temperature': 探索温度
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'Episode/{key}', value, episode)

    def log_training_metrics(self, step: int, metrics: Dict[str, float]):
        """
        记录训练步骤的指标

        Args:
            step: 全局训练步数
            metrics: 指标字典，可包含：
                - 'policy_loss': 策略损失
                - 'value_loss': 价值损失
                - 'total_loss': 总损失
                - 'learning_rate': 学习率
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'Training/{key}', value, step)

    def log_evaluation_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        记录评估指标

        Args:
            episode: 评估时的episode编号
            metrics: 指标字典，可包含：
                - 'avg_score': 平均得分
                - 'max_score': 最高得分
                - 'success_rate': 成功率
                - 'eval_time': 评估时间
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'Evaluation/{key}', value, episode)

    def log_network_weights(self, model: torch.nn.Module, step: int):
        """
        记录网络参数分布

        Args:
            model: PyTorch模型
            step: 全局训练步数
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, step)

    def log_action_distribution(self, action_probs: Dict, episode: int):
        """
        记录动作概率分布

        Args:
            action_probs: 动作概率字典
            episode: 当前episode编号
        """
        if action_probs:
            probs_list = list(action_probs.values())
            self.writer.add_histogram('Policy/ActionProbs',
                                      np.array(probs_list), episode)

    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_dir}")


class CSVLogger:
    """
    CSV格式的训练日志记录器
    用于备份和快速查看训练历史
    """

    def __init__(self, filename: str = None):
        """
        初始化CSV日志记录器

        Args:
            filename: CSV文件名，默认使用时间戳命名
        """
        if filename is None:
            filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.filename = filename
        self.header_written = False
        print(f"CSV logs: {filename}")

    def log_episode(self, episode: int, **kwargs):
        """
        记录一个episode的数据

        Args:
            episode: episode编号
            **kwargs: 要记录的键值对
        """
        import csv

        # 准备数据
        data = {'episode': episode}
        data.update(kwargs)

        # 写入CSV
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerow(data)


class TrainingMonitor:
    """
    综合训练监控器
    同时使用TensorBoard和CSV记录训练过程
    """

    def __init__(self, log_dir: str = None):
        """
        初始化训练监控器

        Args:
            log_dir: 日志根目录
        """
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f"training_logs_{timestamp}"

        os.makedirs(log_dir, exist_ok=True)

        self.tensorboard = TrainingVisualizer(os.path.join(log_dir, 'tensorboard'))
        self.csv_logger = CSVLogger(os.path.join(log_dir, 'metrics.csv'))
        self.start_time = datetime.now()

    def log_training_progress(self, episode: int, step: int,
                              episode_metrics: Dict, training_metrics: Dict):
        """
        记录综合训练进度

        Args:
            episode: 当前episode
            step: 全局训练步数
            episode_metrics: 游戏指标
            training_metrics: 训练指标
        """
        # TensorBoard记录
        self.tensorboard.log_episode_metrics(episode, episode_metrics)
        self.tensorboard.log_training_metrics(step, training_metrics)

        # CSV记录
        combined = {'episode': episode, 'step': step}
        combined.update(episode_metrics)
        combined.update(training_metrics)
        self.csv_logger.log_episode(**combined)

    def close(self):
        """关闭所有记录器"""
        self.tensorboard.close()
        elapsed = datetime.now() - self.start_time
        print(f"\nTraining completed in {elapsed}")
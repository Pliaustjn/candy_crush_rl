import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CandyCrushNet(nn.Module):
    """
    Candy Crush Saga 类三消游戏的神经网络模型。

    输入: (batch_size, 9, 8, 8) - 通道优先的 PyTorch 格式
        通道 0-6: 7 个 one-hot 棋盘状态通道
        通道 7:   步数比例（全局填充）
        通道 8:   得分比例（全局填充）

    输出:
        policy: (batch_size, 112) - 动作概率分布
        value:  (batch_size, 1)   - 状态价值
    """

    def __init__(self, num_filters=64):
        super(CandyCrushNet, self).__init__()

        # 卷积层：输入 9 通道，输出 64 通道，空间尺寸保持 8x8
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=num_filters,
                               kernel_size=3, padding=1)  # padding='same' 等价于 padding=1 for kernel_size=3
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=3, padding=1)

        # 全局平均池化：将 (8, 8, 64) 压缩为 (64,)
        # 使用 AdaptiveAvgPool2d 输出 (1, 1)，然后展平

        # 策略头：全连接层，64 -> 112
        self.policy_head = nn.Linear(num_filters, 112)

        # 价值头：全连接层，64 -> 1
        self.value_head = nn.Linear(num_filters, 1)

    def forward(self, x, legal_actions_mask=None):
        """
        前向传播。

        参数:
            x: 输入张量，形状 (batch_size, 9, 8, 8)
            legal_actions_mask: 合法动作掩码，形状 (batch_size, 112)，1 表示合法，0 表示非法。
                               如果为 None，则不进行屏蔽。

        返回:
            policy: 动作概率分布 (batch_size, 112)
            value:  状态价值 (batch_size, 1)
        """
        # 卷积层，ReLU 激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # 输出形状: (batch_size, 64, 8, 8)

        # 全局平均池化
        # AdaptiveAvgPool2d(1) 将每个通道的空间维度 8x8 平均为 1x1
        h = F.adaptive_avg_pool2d(x, (1, 1))  # 输出形状: (batch_size, 64, 1, 1)
        h = h.view(h.size(0), -1)  # 展平为 (batch_size, 64)

        # 策略头
        policy_logits = self.policy_head(h)  # 输出形状: (batch_size, 112)

        # 对非法动作进行屏蔽：将非法动作的 logit 设为负大数
        if legal_actions_mask is not None:
            # 将非法动作 (mask == 0) 的 logit 替换为 -1e9
            policy_logits = policy_logits.masked_fill(legal_actions_mask == 0, -1e9)

        policy = F.softmax(policy_logits, dim=1)

        # 价值头
        value = self.value_head(h)  # 输出形状: (batch_size, 1)
        value = torch.tanh(value)  # 范围 [-1, 1]

        return policy, value


def create_action_mapping():
    """
    创建动作索引到 (row, col, direction) 的映射。
    共 112 个动作：56 个水平交换 (direction='h') + 56 个垂直交换 (direction='v')

    水平交换: 8 行 × 7 列（每行相邻交换位置 0-1, 1-2, ..., 6-7）
    垂直交换: 7 行 × 8 列（每列相邻交换位置 0-1, 1-2, ..., 6-7）

    返回:
        action_map: list of dict，索引 i 对应动作 {'row': r, 'col': c, 'direction': 'h'/'v'}
    """
    action_map = []

    # 水平交换：交换 (r, c) 和 (r, c+1)，共 8*7 = 56 个
    for r in range(8):
        for c in range(7):
            action_map.append({'row': r, 'col': c, 'direction': 'h'})

    # 垂直交换：交换 (r, c) 和 (r+1, c)，共 7*8 = 56 个
    for r in range(7):
        for c in range(8):
            action_map.append({'row': r, 'col': c, 'direction': 'v'})

    assert len(action_map) == 112
    return action_map


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建网络
    model = CandyCrushNet(num_filters=64)

    # 模拟一个 batch 的输入
    batch_size = 4
    # 创建随机 one-hot 棋盘 (batch_size, 7, 8, 8)
    board_one_hot = torch.zeros(batch_size, 7, 8, 8)
    for b in range(batch_size):
        for r in range(8):
            for c in range(8):
                # 随机选一个糖果类型 (0-5) 或障碍 (6)
                candy_type = np.random.randint(0, 7)
                board_one_hot[b, candy_type, r, c] = 1.0

    # 创建辅助通道
    step_ratio = torch.full((batch_size, 1, 8, 8), 0.3)  # 已用 30% 步数
    score_ratio = torch.full((batch_size, 1, 8, 8), 0.5)  # 得分 50% 目标

    # 拼接输入: (batch_size, 9, 8, 8)
    input_tensor = torch.cat([board_one_hot, step_ratio, score_ratio], dim=1)
    print(f"输入形状: {input_tensor.shape}")  # (4, 9, 8, 8)

    # 模拟合法动作掩码
    legal_mask = torch.ones(batch_size, 112)
    # 将一些动作设为非法（示例：每 batch 随机屏蔽一半动作）
    for b in range(batch_size):
        illegal_indices = np.random.choice(112, size=30, replace=False)
        legal_mask[b, illegal_indices] = 0

    # 前向传播
    policy, value = model(input_tensor, legal_actions_mask=legal_mask)
    print(f"策略输出形状: {policy.shape}")  # (4, 112)
    print(f"价值输出形状: {value.shape}")  # (4, 1)
    print(f"策略概率和: {policy.sum(dim=1)}")  # 应为全 1（合法动作概率和为 1）
    print(f"价值范围: [{value.min().item():.4f}, {value.max().item():.4f}]")  # [-1, 1]

    # 验证非法动作概率为 0
    for b in range(batch_size):
        illegal_probs = policy[b][legal_mask[b] == 0]
        assert torch.all(illegal_probs == 0), f"样本 {b} 存在非法的非零概率动作"
    print("✓ 非法动作概率均为 0")

    # 创建动作映射
    action_map = create_action_mapping()
    print(f"动作总数: {len(action_map)}")
    print(f"动作 0: {action_map[0]}")  # 水平交换 (0,0)-(0,1)
    print(f"动作 56: {action_map[56]}")  # 垂直交换 (0,0)-(1,0)
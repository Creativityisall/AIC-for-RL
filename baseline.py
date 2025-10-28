import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAgent(ABC):
    """
    基类，定义了所有 Agent 的统一接口／行为：
      - 随机种子管理
      - 设备（CPU/CUDA）选择
      - 观测与动作的历史缓存
      - 动作产生的统一流程（reshape → 前向推理 → clip → 缓存 → 返回）
    """

    def __init__(self, seed: int = None):
        """
        参数:
          seed (int, optional): 随机种子；如果不为 None，将同时设置 Python、NumPy、PyTorch 的随机性。
        """
        if seed is not None:
            self.seed(seed)

        # 自动选择运算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 用于记录整个 episode 内的观测和动作（可用于调试或训练数据收集）
        self.obs_history = []
        self.act_history = []

    def seed(self, seed: int = 123) -> None:
        """设置随机种子，保证结果可复现。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        """
        在每个 episode 开始时由外部调用，
        清空历史缓存。
        """
        self.obs_history.clear()
        self.act_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        统一的动作生成流程：
          1. 将观测拉平成一维向量
          2. 调用子类的 get_action() 得到原始动作
          3. 将动作限制到 [-1, 1]
          4. 缓存本次观测和动作
          5. 返回动作
        """
        obs = obs.reshape(-1).astype(np.float32)
        action = self.get_action(obs)
        action = np.clip(action, -1.0, 1.0).reshape(-1).astype(np.float32)

        self.obs_history.append(obs)
        self.act_history.append(action)

        return action

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        子类必须实现：根据当前观测 obs 计算动作。
        输入:
          obs (np.ndarray): 一维观测向量
        输出:
          action (np.ndarray): 一维动作向量
        """
        ...

    def close(self) -> None:
        """可选：释放资源（如环境连接、文件句柄等）。"""
        pass


class Mlp(nn.Module):
    """
    简单三层全连接网络 (MLP)：
      - 输入层 → 隐藏层1 (ReLU)
      - 隐藏层1 → 隐藏层2 (ReLU)
      - 隐藏层2 → 输出层（线性）
    """

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        """
        参数:
          input_dim  (int):  输入特征维度
          hidden_dim1(int): 第一隐藏层单元数
          hidden_dim2(int): 第二隐藏层单元数
          output_dim (int):  输出维度（动作维度）
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算：
          x → ReLU(fc1) → ReLU(fc2) → fc3 → 返回
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyAgent_19(BaseAgent):
    """
    基于预训练 MLP 的策略 Agent。
    维护最近 30 帧的观测历史，将它们拼接后送入网络，
    输出当前动作。
    """

    HIST_LEN = 30  # 使用的历史观测帧数

    def __init__(self, model_name):
        """
        初始化时加载预训练模型，将网络切换到 eval 模式并移动到指定设备。
        """
        super().__init__()

        # 自动定位模型文件（与当前脚本同目录下）
        model_path = os.path.join(os.path.dirname(__file__), model_name)

        # 网络结构与保存的权重需保持一致
        # 注意这里的维度，这个模型是直接一次看30步。
        self.policy = Mlp(input_dim=150, hidden_dim1=256, hidden_dim2=256, output_dim=3)
        state_dict = torch.load(model_path, map_location="cpu")
        self.policy.load_state_dict(state_dict)

        self.policy.to(self.device)
        self.policy.eval()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前观测和历史缓存构造网络输入并前向推理：
          1. 将历史 obs + 当前 obs 合并成列表
          2. 不足 30 帧时用最早那帧进行填充；超过 30 帧则取最新 30 帧
          3. 将帧序列倒序、拼成一维 state
          4. 转为 Tensor，添加 batch 维度并移动到 device
          5. 无梯度前向，输出动作
          6. 去 batch 维度、转回 Numpy 并返回
        """
        seq = self.obs_history + [obs]
        if len(seq) < self.HIST_LEN:
            # 不足时用第一帧填充到指定长度
            seq = [seq[0]] * (self.HIST_LEN - len(seq)) + seq
        else:
            seq = seq[-self.HIST_LEN :]

        # 倒序并拉平成一维
        state = np.array(seq[::-1], dtype=np.float32).reshape(1, -1)

        # 转 Tensor，前向推理
        x = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            out = self.policy(x)

        # 去除 batch 维度，返回 numpy 数组
        return out.squeeze(0).cpu().numpy()

class PolicyAgent_18(BaseAgent):
    """
    基于预训练 MLP 的策略 Agent。
    维护最近 30 帧的观测历史，将它们拼接后送入网络，
    输出当前动作。
    """

    HIST_LEN = 30  # 使用的历史观测帧数

    def __init__(self, model_name):
        """
        初始化时加载预训练模型，将网络切换到 eval 模式并移动到指定设备。
        """
        super().__init__()

        # 自动定位模型文件（与当前脚本同目录下）
        model_path = os.path.join(os.path.dirname(__file__), model_name)

        # 网络结构与保存的权重需保持一致
        # 注意这里的维度，这个模型是直接一次看30步。
        self.policy = Mlp(input_dim=150, hidden_dim1=256, hidden_dim2=128, output_dim=3)
        state_dict = torch.load(model_path, map_location="cpu")
        self.policy.load_state_dict(state_dict)

        self.policy.to(self.device)
        self.policy.eval()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前观测和历史缓存构造网络输入并前向推理：
          1. 将历史 obs + 当前 obs 合并成列表
          2. 不足 30 帧时用最早那帧进行填充；超过 30 帧则取最新 30 帧
          3. 将帧序列倒序、拼成一维 state
          4. 转为 Tensor，添加 batch 维度并移动到 device
          5. 无梯度前向，输出动作
          6. 去 batch 维度、转回 Numpy 并返回
        """
        seq = self.obs_history + [obs]
        if len(seq) < self.HIST_LEN:
            # 不足时用第一帧填充到指定长度
            seq = [seq[0]] * (self.HIST_LEN - len(seq)) + seq
        else:
            seq = seq[-self.HIST_LEN :]

        # 倒序并拉平成一维
        state = np.array(seq[::-1], dtype=np.float32).reshape(1, -1)

        # 转 Tensor，前向推理
        x = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            out = self.policy(x)

        # 去除 batch 维度，返回 numpy 数组
        return out.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    agent = PolicyAgent()
    dummy_obs = np.random.rand(150).astype(np.float32)

    a1 = agent.act(dummy_obs)
    a2 = agent.act(dummy_obs * 2)

    print("Action at step 1:", a1)
    print("Action at step 2:", a2)

    agent.reset()
    agent.close()
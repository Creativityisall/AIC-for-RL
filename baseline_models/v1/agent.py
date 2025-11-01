import os
import random
from abc import ABC, abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ---------------------------------
# --- Global Parameters (與訓練時保持一致) ---
# ---------------------------------
HIST_LEN = 30
EMBED_DIM = 256
NHEAD = 4
NUM_LAYERS = 4
MLP_HIDDEN = 256
OBS_DIM = 5
ACTION_DIM = 3

# --- 模型和數據路徑 ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "v1.2")  # 假設模型文件與此腳本在同一目錄下
IQL_ACTOR_PATH = os.path.join(MODEL_DIR, "industrial_iql_actor.pth")
IQL_EMBED_PATH = os.path.join(MODEL_DIR, "industrial_iql_embed.pth")
NORMALIZER_MEAN_PATH = os.path.join(MODEL_DIR, "normalizer_mean.npy")
NORMALIZER_STD_PATH = os.path.join(MODEL_DIR, "normalizer_std.npy")


# -----------------------------------------------------------
# --- 1. 從訓練代碼中複製必要的網絡定義 ---
# -----------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequenceEmbedding(nn.Module):
    def __init__(self, obs_dim, hist_len, embed_dim, nhead, num_layers):
        super().__init__()
        self.input_embed = nn.Linear(obs_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=hist_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=MLP_HIDDEN,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_embed(obs_seq)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        return transformer_out[:, -1, :]


def build_mlp(input_dim, output_dim, hidden_units=[256, 256]):
    layers = [nn.Linear(input_dim, hidden_units[0]), nn.ReLU()]
    for i in range(len(hidden_units) - 1):
        layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_units[-1], output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.net = build_mlp(embed_dim, action_dim)

    def forward(self, embed):
        return torch.tanh(self.net(embed))


# -----------------------------------------------------------
# --- 2. 您提供的 BaseAgent 框架 ---
# -----------------------------------------------------------
class BaseAgent(ABC):
    """
    基类，定义了所有 Agent 的统一接口／行为：
      - 随机种子管理
      - 设备（CPU/CUDA）选择
      - 观测与动作的历史缓存
      - 动作产生的统一流程（reshape → 前向推理 → clip → 缓存 → 返回）
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            self.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_history = []
        self.act_history = []

    def seed(self, seed: int = 123) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        self.obs_history.clear()
        self.act_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(-1).astype(np.float32)
        action = self.get_action(obs)
        action = np.clip(action, -1.0, 1.0).reshape(-1).astype(np.float32)
        self.obs_history.append(obs)
        self.act_history.append(action)
        return action

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        pass


# -----------------------------------------------------------
# --- 3. 實現 PolicyAgent (原 IQLAgent) ---
# -----------------------------------------------------------
# 【關鍵修改】: 將類別名稱從 IQLAgent 更改為 PolicyAgent 以匹配 evaluator.py
class PolicyAgent(BaseAgent):
    """
    基於預訓練的 History-Aware IQL 模型的智能體。
    - 加載序列嵌入網絡和 Actor 網絡。
    - 加載訓練時計算的均值和標準差，用於歸一化輸入。
    - 維護最近 30 幀的觀測歷史，並在推理時使用。
    """

    def __init__(self):
        super().__init__()
        print(f"Initializing PolicyAgent (IQL-based) on device: {self.device}")

        # 1. 加載 Normalizer 的統計數據
        try:
            self.obs_mean = np.load(NORMALIZER_MEAN_PATH)
            self.obs_std = np.load(NORMALIZER_STD_PATH)
            print("✅ Successfully loaded normalizer mean and std.")
            print(f"   - Mean shape: {self.obs_mean.shape}")
            print(f"   - Std shape: {self.obs_std.shape}")
        except FileNotFoundError as e:
            print(f"❌ Error: Normalizer file not found. Make sure to save them during training. {e}")
            raise

        # 2. 初始化網絡結構
        self.embed_net = SequenceEmbedding(OBS_DIM, HIST_LEN, EMBED_DIM, NHEAD, NUM_LAYERS)
        self.actor = Actor(EMBED_DIM, ACTION_DIM)

        # 3. 加載預訓練權重
        try:
            load_kwargs = {'map_location': self.device}
            self.embed_net.load_state_dict(torch.load(IQL_EMBED_PATH, weights_only=True, **load_kwargs))
            self.actor.load_state_dict(torch.load(IQL_ACTOR_PATH, weights_only=True, **load_kwargs))
            print("✅ Successfully loaded pretrained embedding and actor models.")
        except FileNotFoundError as e:
            print(f"❌ Error: Model file not found. Check paths: {IQL_EMBED_PATH} or {IQL_ACTOR_PATH}. {e}")
            raise

        # 4. 將網絡移動到指定設備並切換到評估模式
        self.embed_net.to(self.device)
        self.actor.to(self.device)
        self.embed_net.eval()
        self.actor.eval()
        print("🚀 Agent is ready.")

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """使用加載的均值和標準差來歸一化觀測值。"""
        # 增加一個小的 epsilon 防止除以零
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根據當前觀測和歷史緩存構造網絡輸入並進行前向推理。
        """
        # 1. 構建包含當前觀測的歷史序列 (使用未歸一化的原始觀測)
        # BaseAgent 的 obs_history 已經存儲了過去的觀測
        seq_unnormalized = self.obs_history + [obs]

        # 2. 填充或截斷歷史序列以滿足 HIST_LEN
        if len(seq_unnormalized) < HIST_LEN:
            # 當歷史不足時，使用最早的一幀進行填充
            padding = [seq_unnormalized[0]] * (HIST_LEN - len(seq_unnormalized))
            obs_seq_unnormalized = np.array(padding + seq_unnormalized, dtype=np.float32)
        else:
            # 當歷史充足時，取最新的 HIST_LEN 幀
            obs_seq_unnormalized = np.array(seq_unnormalized[-HIST_LEN:], dtype=np.float32)

        # 3. 【關鍵】對整個序列進行歸一化
        obs_seq_normalized = self._normalize_obs(obs_seq_unnormalized)

        # 4. 轉換爲 Tensor，添加 batch 維度並移動到 device
        obs_tensor = torch.from_numpy(obs_seq_normalized).unsqueeze(0).to(self.device)

        # 5. 無梯度前向推理
        with torch.no_grad():
            # (1, HIST_LEN, OBS_DIM) -> (1, EMBED_DIM)
            state_embedding = self.embed_net(obs_tensor)
            # (1, EMBED_DIM) -> (1, ACTION_DIM)
            action_tensor = self.actor(state_embedding)

        # 6. 去除 batch 維度，轉回 Numpy 數組並返回
        return action_tensor.squeeze(0).cpu().numpy()


# -----------------------------------------------------------
# --- 4. 主程序入口：演示如何使用 PolicyAgent ---
# -----------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing PolicyAgent (IQL-based) ---")

    # 在運行此腳本前，請確保以下文件存在於同級目錄下：
    # - industrial_iql_actor.pth
    # - industrial_iql_embed.pth
    # - normalizer_mean.npy
    # - normalizer_std.npy
    # 如果文件不存在，可以先創建僞造的佔位文件來測試代碼結構。

    try:
        # 實例化智能體
        agent = PolicyAgent() # <-- 名稱已更新

        # 重置智能體狀態 (清空歷史記錄)
        agent.reset()

        # 模擬一個 episode，共 50 步
        print("\n--- Simulating one episode (50 steps) ---")
        for i in range(50):
            # 創建一個符合維度的隨機觀測
            # 在實際應用中，這將來自於您的環境 env.step()
            dummy_obs = np.random.rand(OBS_DIM).astype(np.float32)

            # 讓智能體根據觀測採取行動
            action = agent.act(dummy_obs)

            print(f"Step {i + 1:02d} | Obs shape: {dummy_obs.shape} -> Action: {action}")
            # 打印歷史長度，以驗證其增長和截斷
            if (i + 1) % 10 == 0:
                print(f"         (Current obs_history length: {len(agent.obs_history)})")

        print("\n--- Simulation finished ---")

        # 清理資源
        agent.close()

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nCould not run simulation due to an error: {e}")
        print(
            "Please make sure all required model and normalizer files are present in the same directory as this script.")
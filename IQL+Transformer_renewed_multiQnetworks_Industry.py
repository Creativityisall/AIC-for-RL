import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
import math
from typing import Tuple, List
import random
import copy

# 【新增】导入学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------
# --- Global Parameters ---
# ---------------------------------
# 【修改】: 根据您的数据预处理逻辑，历史长度为30
HIST_LEN = 30
EMBED_DIM = 256
NHEAD = 4
NUM_LAYERS = 4
MLP_HIDDEN = 256

# --- Training Hyperparameters ---
EVAL_EPISODES = 10  # 虽然没有在线环境，但保留参数以备将来扩展
# 【修改】: 使用新的模型保存路径
IQL_ACTOR_PATH = "agent/industrial_iql_actor.pth"
IQL_CRITIC_PATH = "agent/industrial_iql_critic.pth"
IQL_VF_PATH = "agent/industrial_iql_vf.pth"
IQL_EMBED_PATH = "agent/industrial_iql_embed.pth"
IQL_ACTOR_OPTIMIZER_PATH = "agent/industrial_iql_actor_optimizer.pth"
IQL_CRITIC_OPTIMIZER_PATH = "agent/industrial_iql_critic_optimizer.pth"
IQL_VF_OPTIMIZER_PATH = "agent/industrial_iql_vf_optimizer.pth"

LOAD_MODELS = True

# --- 核心维度 ---
# 【修改】: 根据您的数据集设置维度
OBS_DIM = 5
ACTION_DIM = 3


# 【新增】 Observation Normalizer 类
# ---------------------------------
class Normalizer:
    """
    根据离线数据集中的 `obs` 和 `next_obs` 计算全局的均值和标准差，
    并用它们来标准化观测值。
    """

    def __init__(self, data: dict):
        all_obs = np.concatenate([data['obs'], data['next_obs']], axis=0).astype(np.float32)
        self.mean = np.mean(all_obs, axis=0)
        self.std = np.std(all_obs, axis=0)
        self.std[self.std < 1e-5] = 1e-5
        print("✅ Normalizer initialized.")
        print(f"  - Mean shape: {self.mean.shape}")
        print(f"  - Std shape: {self.std.shape}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std


# ---------------------------------
# --- 1. 序列嵌入网络 (Transformer Encoder) ---
# ---------------------------------
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
        print(f"[SequenceEmbedding] Initialized: Obs({obs_dim}) -> Embed({embed_dim})")

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_embed(obs_seq)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        return transformer_out[:, -1, :]


# ---------------------------------
# --- 2. IQL 模型定义 ---
# ---------------------------------
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
        print(f"[Actor] MLP initialized: Input({embed_dim}) -> Output({action_dim})")

    def forward(self, embed):
        return torch.tanh(self.net(embed))


class Critic(nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.net = build_mlp(embed_dim + action_dim, 1)
        print(f"[Critic] MLP initialized: Input({embed_dim + action_dim}) -> Output(1)")

    def forward(self, embed, action):
        return self.net(torch.cat([embed, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = build_mlp(embed_dim, 1)
        print(f"[ValueFunction] MLP initialized: Input({embed_dim}) -> Output(1)")

    def forward(self, embed):
        return self.net(embed).squeeze(-1)


# ---------------------------------
# --- 3. 序列化数据集 ---
# ---------------------------------
class SequenceRLDataset(Dataset):
    def __init__(self, data: dict, hist_len: int, obs_dim: int, action_dim: int, normalizer: Normalizer):
        self.obs = normalizer.normalize(data['obs'].astype(np.float32))
        self.next_obs = normalizer.normalize(data['next_obs'].astype(np.float32))
        self.act = data['action'].astype(np.float32)
        self.rew = data['reward'].astype(np.float32).squeeze()
        self.done = data['done'].astype(np.float32).squeeze()
        self.hist_len = hist_len

        if self.obs.shape[-1] != obs_dim or self.act.shape[-1] != action_dim:
            print(
                f"❌ 警告: 数据集维度不匹配! Config Obs={obs_dim}, Got Obs={self.obs.shape[-1]}; Config Act={action_dim}, Got Act={self.act.shape[-1]}")

        self.total_size = len(self.obs)
        self.start_idx = hist_len - 1
        self.available_samples = self.total_size - self.start_idx - 1

        print(f"✅ 成功加载离线数据集 (已标准化)，总数据点: {self.total_size}, 可用训练样本: {self.available_samples}")
        if self.available_samples <= 0:
            raise ValueError(f"数据集过小 (可用样本: {self.available_samples})，无法形成长度为 {hist_len} 的序列。")

    def __len__(self):
        return self.available_samples

    def __getitem__(self, idx: int):
        t = idx + self.start_idx
        obs_seq = torch.from_numpy(self.obs[t - self.hist_len + 1: t + 1])
        next_obs_seq = torch.from_numpy(self.obs[t - self.hist_len + 2: t + 2])
        action_t = torch.from_numpy(self.act[t])
        reward_t = torch.tensor(self.rew[t])
        done_t = torch.tensor(self.done[t])
        return obs_seq, next_obs_seq, action_t, reward_t, done_t


# ---------------------------------
# --- 4. IQL 训练逻辑 ---
# ---------------------------------
def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return (weight * (diff ** 2)).mean()


def huber_loss(diff, delta: float = 1.0):
    abs_diff = torch.abs(diff)
    quadratic_part = (1.0 / (2.0 * delta)) * (diff ** 2)
    linear_part = abs_diff - 0.5 * delta
    loss = torch.where(abs_diff <= delta, quadratic_part, linear_part)
    return loss.mean()


# 【修改】: 增加 schedulers 参数
def train_one_epoch_iql(models, optimizers, loader, device, iql_tau, iql_beta, discount, embed_net, schedulers=None):
    actor, critic1, critic2, vf, target_critic1, target_critic2 = models
    actor_optimizer, critic_optimizer, vf_optimizer = optimizers

    # 【新增】: 解包调度器
    if schedulers is not None:
        actor_scheduler, critic_scheduler, vf_scheduler = schedulers

    actor.train();
    critic1.train();
    critic2.train();
    vf.train();
    embed_net.train()
    total_vf_loss, total_critic_loss, total_actor_loss = 0.0, 0.0, 0.0

    for obs_seq, next_obs_seq, act, rew, done in loader:
        obs_seq, next_obs_seq, act, rew, done = (
            obs_seq.to(device), next_obs_seq.to(device), act.to(device), rew.to(device), done.to(device)
        )
        current_embed = embed_net(obs_seq)
        with torch.no_grad():
            next_embed = embed_net(next_obs_seq)

        # 1. 训练 Value Function (VF)
        with torch.no_grad():
            q1_target_pred = target_critic1(current_embed.detach(), act)
            q2_target_pred = target_critic2(current_embed.detach(), act)
            q_target_for_vf_and_actor = torch.min(q1_target_pred, q2_target_pred)
        vf_pred = vf(current_embed)
        vf_loss = expectile_loss(q_target_for_vf_and_actor - vf_pred, expectile=iql_tau)
        vf_optimizer.zero_grad();
        vf_loss.backward(retain_graph=True);
        vf_optimizer.step()
        total_vf_loss += vf_loss.item()

        # 2. 训练 Critic (Q Function)
        with torch.no_grad():
            next_v = vf(next_embed)
            q_target_for_critic = rew + (1.0 - done) * discount * next_v
        q1_pred = critic1(current_embed.detach(), act)
        q2_pred = critic2(current_embed.detach(), act)
        critic_loss = huber_loss(q1_pred - q_target_for_critic) + huber_loss(q2_pred - q_target_for_critic)
        critic_optimizer.zero_grad();
        critic_loss.backward(retain_graph=True);
        critic_optimizer.step()
        total_critic_loss += critic_loss.item()

        # 3. 训练 Actor (Policy)
        with torch.no_grad():
            advantage = q_target_for_vf_and_actor - vf_pred.detach()
            exp_advantage = torch.exp(iql_beta * advantage).clamp(max=100.0)#TODO:打印一下看看这里的Q一般是多少，确定这里的clamp取多少合理。
        policy_out = actor(current_embed)
        actor_loss = (exp_advantage * F.mse_loss(policy_out, act, reduction='none').sum(dim=-1)).mean()
        actor_optimizer.zero_grad();
        actor_loss.backward();
        actor_optimizer.step()
        total_actor_loss += actor_loss.item()

    num_batches = len(loader)

    # 【新增】: 在每个 Epoch 结束时更新学习率
    if schedulers is not None:
        actor_scheduler.step()
        critic_scheduler.step()
        vf_scheduler.step()

    return total_vf_loss / num_batches, total_critic_loss / num_batches, total_actor_loss / num_batches


# 【修改】: run_iql_training 中新增调度器初始化和学习率重置逻辑
def run_iql_training(data: dict, device: torch.device):
    obs_dim, action_dim = OBS_DIM, ACTION_DIM

    normalizer = Normalizer(data)

    # -----------------------------------------------------------------
    # --- 保存 Normalizer 的 mean 和 std ---
    # -----------------------------------------------------------------
    agent_dir = os.path.dirname(IQL_ACTOR_PATH)
    os.makedirs(agent_dir, exist_ok=True)
    mean_path = os.path.join(agent_dir, "normalizer_mean.npy")
    std_path = os.path.join(agent_dir, "normalizer_std.npy")
    np.save(mean_path, normalizer.mean)
    np.save(std_path, normalizer.std)
    print(f"✅ Normalizer mean and std saved successfully to '{agent_dir}'.")
    # -----------------------------------------------------------------

    dataset = SequenceRLDataset(data, hist_len=HIST_LEN, obs_dim=obs_dim, action_dim=action_dim, normalizer=normalizer)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, _ = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"数据集划分: {train_size} 训练 / {val_size} 验证 (验证集仅用于监控，不在此范例中使用)")

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    iql_tau, iql_beta, discount = 0.92, 3.0, 0.99
    # epochs=200 用于 CosineAnnealingLR 的 T_max
    # 【注意】: learning_rate 是我们强制重置的学习率起始值
    target_update_rate, epochs, learning_rate = 0.002, 100, 4e-5

    embed_net = SequenceEmbedding(obs_dim, HIST_LEN, EMBED_DIM, NHEAD, NUM_LAYERS).to(device)
    actor = Actor(EMBED_DIM, action_dim).to(device)
    critic1 = Critic(EMBED_DIM, action_dim).to(device)
    critic2 = Critic(EMBED_DIM, action_dim).to(device)
    vf = ValueFunction(EMBED_DIM).to(device)
    target_critic1, target_critic2 = copy.deepcopy(critic1), copy.deepcopy(critic2)

    # 1. 定义优化器
    actor_optimizer = optim.Adam(list(actor.parameters()) + list(embed_net.parameters()), lr=learning_rate)
    critic_optimizer = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=learning_rate)
    vf_optimizer = optim.Adam(vf.parameters(), lr=learning_rate)

    # 2. 定义学习率调度器 (使用 CosineAnnealingLR)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=epochs)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=epochs)
    vf_scheduler = CosineAnnealingLR(vf_optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(IQL_ACTOR_PATH), exist_ok=True)
    if LOAD_MODELS:
        try:
            load_kwargs = {'map_location': device}
            # 加载模型
            actor.load_state_dict(torch.load(IQL_ACTOR_PATH, **load_kwargs))
            critic1.load_state_dict(torch.load(IQL_CRITIC_PATH.replace(".pth", "_1.pth"), **load_kwargs))
            critic2.load_state_dict(torch.load(IQL_CRITIC_PATH.replace(".pth", "_2.pth"), **load_kwargs))
            vf.load_state_dict(torch.load(IQL_VF_PATH, **load_kwargs))
            embed_net.load_state_dict(torch.load(IQL_EMBED_PATH, **load_kwargs))
            target_critic1.load_state_dict(critic1.state_dict())
            target_critic2.load_state_dict(critic2.state_dict())
            print(f"✅ 成功加载预训练 IQL 模型")

            # 加载优化器状态 (会继承旧的极小学习率)
            actor_optimizer.load_state_dict(torch.load(IQL_ACTOR_OPTIMIZER_PATH, map_location=device))
            critic_optimizer.load_state_dict(torch.load(IQL_CRITIC_OPTIMIZER_PATH, map_location=device))
            vf_optimizer.load_state_dict(torch.load(IQL_VF_OPTIMIZER_PATH, map_location=device))
            print("✅ 成功加载所有优化器状态 (但学习率可能极小)")

            # -----------------------------------------------------------------
            # 【核心修改】：强制重置学习率 (覆盖继承的极小值)
            # -----------------------------------------------------------------
            actor_optimizer.param_groups[0]['lr'] = learning_rate
            critic_optimizer.param_groups[0]['lr'] = learning_rate
            vf_optimizer.param_groups[0]['lr'] = learning_rate
            print(f"🔄 强制重置优化器学习率到新的起始值: {learning_rate:.2e}。调度器将从此值开始衰减。")
            # -----------------------------------------------------------------

        except Exception as e:
            print(f"⚠️ 加载模型或优化器失败 ({e})，将从头开始训练。")

    models = (actor, critic1, critic2, vf, target_critic1, target_critic2)
    optimizers = (actor_optimizer, critic_optimizer, vf_optimizer)

    # 3. 将调度器打包传递
    schedulers = (actor_scheduler, critic_scheduler, vf_scheduler)

    print(f"--- 开始 History-Aware IQL 训练 (数据源: IndustrialControl) ---")
    best_train_actor_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # 传递 schedulers 参数
        vf_loss, critic_loss, actor_loss = train_one_epoch_iql(
            models, optimizers, train_loader, device, iql_tau, iql_beta, discount, embed_net, schedulers=schedulers
        )

        # 目标网络软更新 (Target Network Soft Update)
        for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
            target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)
        for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
            target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)

        if actor_loss < best_train_actor_loss:
            best_train_actor_loss = actor_loss
            # 模型和优化器保存
            torch.save(actor.state_dict(), IQL_ACTOR_PATH)
            torch.save(critic1.state_dict(), IQL_CRITIC_PATH.replace(".pth", "_1.pth"))
            torch.save(critic2.state_dict(), IQL_CRITIC_PATH.replace(".pth", "_2.pth"))
            torch.save(vf.state_dict(), IQL_VF_PATH)
            torch.save(embed_net.state_dict(), IQL_EMBED_PATH)
            torch.save(actor_optimizer.state_dict(), IQL_ACTOR_OPTIMIZER_PATH)
            torch.save(critic_optimizer.state_dict(), IQL_CRITIC_OPTIMIZER_PATH)
            torch.save(vf_optimizer.state_dict(), IQL_VF_OPTIMIZER_PATH)
            print(f"  -> Epoch {epoch:03d} 新的最佳模型已保存 (Actor Loss: {best_train_actor_loss:.6f})")

        if epoch % 5 == 0 or epoch == 1:
            # 打印当前学习率以供监控 (此时已是经过 scheduler.step() 后的学习率)
            current_lr = actor_optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:03d} | LR: {current_lr:.2e} | VF Loss: {vf_loss:.6f} | Critic Loss: {critic_loss:.6f} | Actor Loss: {actor_loss:.6f}")

    print(f"\n--- 训练结束，最佳训练 Actor Loss: {best_train_actor_loss:.6f} ---")


# ---------------------------------
# --- Main Execution Block ---
# ---------------------------------
if __name__ == '__main__':
    print(f"--- 正在从 'IndustrialControl-main/data/data2.csv' 初始化离线数据集 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 注意: 这里的路径假设 data.csv 位于 IndustrialControl-main/data/ 下
        data = pd.read_csv("IndustrialControl-main/data/data2.csv")

        obs_list, action_list, next_obs_list, reward_list, done_list = [], [], [], [], []

        for idx in data['index'].unique():
            traj_data = data[data['index'] == idx].reset_index(drop=True)
            obs_cols = [f'obs_{i}' for i in range(1, 6)]
            action_cols = [f'action_{i}' for i in range(1, 4)]
            obs = traj_data[obs_cols].values
            actions = traj_data[action_cols].values
            num_steps = len(obs)

            obs_list.append(obs[:-1])
            action_list.append(actions[:-1])
            next_obs_list.append(obs[1:])
            reward_list.append(np.zeros(num_steps - 1))
            dones = np.zeros(num_steps - 1)
            dones[-1] = 1.0
            done_list.append(dones)

        final_obs = np.concatenate(obs_list, axis=0)
        final_actions = np.concatenate(action_list, axis=0)
        final_next_obs = np.concatenate(next_obs_list, axis=0)
        final_rewards = np.concatenate(reward_list, axis=0)
        final_dones = np.concatenate(done_list, axis=0)

        train_data = {
            'obs': np.concatenate([final_obs, final_next_obs[-1:]], axis=0),
            'action': final_actions,
            'next_obs': final_next_obs,
            'reward': final_rewards,
            'done': final_dones
        }

        print(f"✅ 成功处理数据，总共 {len(final_obs)} 个转换。")

    except FileNotFoundError:
        print(f"❌ 错误: 未找到数据文件 'IndustrialControl-main/data/data.csv'。请确保文件路徑正确。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 处理数据时发生错误: {e}")
        sys.exit(1)

    run_iql_training(train_data, device)
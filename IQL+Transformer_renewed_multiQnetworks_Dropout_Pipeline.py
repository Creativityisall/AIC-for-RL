import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import neorl2
import gymnasium as gym
import sys
import math
from typing import Tuple, List
import random
import copy

# ---------------------------------
# --- Global Parameters ---
# ---------------------------------
HIST_LEN = 30
EMBED_DIM = 256
NHEAD = 4
NUM_LAYERS = 4
MLP_HIDDEN = 256

# --- Training Hyperparameters ---
EVAL_EPISODES = 10
N_CRITICS = 6
ALPHA_QUANTILE = 0.25
EMBED_DROPOUT_P = 0.2  # 【新增】嵌入层丢弃的概率 (p)

IQL_ACTOR_PATH = "agent/iql_actor_modelbeta=1withtricks.pth"
IQL_CRITIC_PATH = "agent/iql_critic_modelbeta=withtricks.pth"
IQL_VF_PATH = "agent/iql_vf_modelbeta=1withtricks.pth"
IQL_EMBED_PATH = "agent/iql_embed_modelbeta=1withtricks.pth"
IQL_ACTOR_OPTIMIZER_PATH = "agent/iql_actor_optimizerbeta=1withtricks.pth"
IQL_CRITIC_OPTIMIZER_PATH = "agent/iql_critic_optimizerbeta=1withtricks.pth"
IQL_VF_OPTIMIZER_PATH = "agent/iql_vf_optimizerbeta=1withtricks.pth"

LOAD_MODELS = False

# --- 核心维度 ---
OBS_DIM = 52
ACTION_DIM = 1


# 【新增】 Observation Normalizer 类
# ---------------------------------
class Normalizer:
    """
    根据图片中的公式，计算并应用 Observation 标准化
    μo = 1/(2N) * Σ(s_i + s'_i)
    σo^2 = 1/(2N) * Σ[(s_i - μo)^2 + (s'_i - μo)^2]
    """

    def __init__(self, data: dict):
        # 将 obs 和 next_obs 拼接在一起，计算全局的均值和标准差
        all_obs = np.concatenate([data['obs'], data['next_obs']], axis=0).astype(np.float32)

        self.mean = np.mean(all_obs, axis=0)
        self.std = np.std(all_obs, axis=0)

        # 添加一个极小值 epsilon 防止除以零
        self.std[self.std < 1e-5] = 1e-5

        print("✅ Normalizer initialized.")
        print(f"  - Mean shape: {self.mean.shape}")
        print(f"  - Std shape: {self.std.shape}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """对输入的 observation 进行标准化"""
        return (obs - self.mean) / self.std

    def to_device(self, device: torch.device):
        """将 mean 和 std 转换为 tensor 并移动到指定设备"""
        self.mean = torch.from_numpy(self.mean).float().to(device)
        self.std = torch.from_numpy(self.std).float().to(device)
        return self


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
    # 【修改】增加 dropout_p 参数
    def __init__(self, obs_dim, hist_len, embed_dim, nhead, num_layers, dropout_p: float):
        super().__init__()
        self.input_embed = nn.Linear(obs_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=hist_len)

        # 【修改】根据论文描述，在此处添加 Embedding Dropout
        self.embed_dropout = nn.Dropout(p=dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=MLP_HIDDEN,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        print(f"[SequenceEmbedding] Initialized: Obs({obs_dim}) -> Embed({embed_dim})")
        print(f"  -> Applying Embedding Dropout with p={dropout_p}")

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # 1. 线性投影 (Linear Projection)
        x = self.input_embed(obs_seq)
        # 2. 添加时间步/位置嵌入 (Time-step/Positional Embedding)
        x = self.pos_encoder(x)
        # 3. 【修改】应用随机维度丢弃 (Randomized Dimension Dropping)
        x = self.embed_dropout(x)
        # 4. 输入 Transformer 主干网络
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
# --- 3. 序列化数据集 (【修改】加入标准化) ---
# ---------------------------------
class SequenceRLDataset(Dataset):
    def __init__(self, data: dict, hist_len: int, obs_dim: int, action_dim: int,
                 normalizer: Normalizer):  # 【修改】接收 normalizer

        # 【修改】使用 normalizer 对数据进行标准化
        self.obs = normalizer.normalize(data['obs'].astype(np.float32))
        self.next_obs = normalizer.normalize(data['next_obs'].astype(np.float32))

        self.act = data['action'].astype(np.float32)
        self.rew = data['reward'].astype(np.float32).squeeze()
        self.done = data['done'].astype(np.float32).squeeze()
        self.hist_len = hist_len

        if self.obs.shape[-1] != obs_dim or self.act.shape[-1] != action_dim:
            print(
                f"❌ 警告: 数据集维度不匹配! 数据集Obs={self.obs.shape[-1]}, Config Obs={obs_dim}; 数据集Act={self.act.shape[-1]}, Config Act={action_dim}")

        self.total_size = len(self.obs)
        self.start_idx = hist_len - 1
        self.available_samples = self.total_size - self.start_idx - 1

        print(f"✅ 成功加载离线数据集 (已标准化)，总数据点: {self.total_size}, 可用训练样本: {self.available_samples}")

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


def train_one_epoch_iql(models, optimizers, loader, device, iql_tau, iql_beta, discount, embed_net, alpha_quantile):
    actor, critics, vf, target_critics = models
    actor_optimizer, critic_optimizer, vf_optimizer = optimizers

    actor.train()
    vf.train()
    embed_net.train()
    for critic in critics:
        critic.train()

    total_vf_loss, total_critic_loss, total_actor_loss = 0.0, 0.0, 0.0

    for obs_seq, next_obs_seq, act, rew, done in loader:
        obs_seq, next_obs_seq, act, rew, done = (
            obs_seq.to(device), next_obs_seq.to(device), act.to(device), rew.to(device), done.to(device)
        )

        current_embed = embed_net(obs_seq)
        with torch.no_grad():
            next_embed = embed_net(next_obs_seq)

        # --- 1. 训练价值网络 (Value Function) ---
        with torch.no_grad():
            all_q_target_preds = torch.stack([tc(current_embed.detach(), act) for tc in target_critics], dim=0)
            q_target_for_vf_and_actor = torch.quantile(all_q_target_preds, q=alpha_quantile, dim=0)

        vf_pred = vf(current_embed)
        vf_err = q_target_for_vf_and_actor - vf_pred
        vf_loss = expectile_loss(vf_err, expectile=iql_tau)

        vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        vf_optimizer.step()
        total_vf_loss += vf_loss.item()

        # --- 2. 训练 Q 值网络 (Critic) ---
        with torch.no_grad():
            next_v = vf(next_embed)
            q_target_for_critic = rew + (1.0 - done) * discount * next_v

        critic_loss = 0.0
        delta_for_huber = 1.0
        for critic in critics:
            q_pred = critic(current_embed, act)
            critic_loss += huber_loss(q_pred - q_target_for_critic, delta=delta_for_huber)

        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_optimizer.step()
        total_critic_loss += critic_loss.item()

        # --- 3. 训练策略网络 (Actor) ---
        with torch.no_grad():
            advantage = q_target_for_vf_and_actor - vf_pred.detach()
            exp_advantage = torch.exp(iql_beta * advantage).clamp(max=100.0)

        policy_out = actor(current_embed)
        actor_loss = (exp_advantage * F.mse_loss(policy_out, act, reduction='none')).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        total_actor_loss += actor_loss.item()

    num_batches = len(loader)
    return total_vf_loss / num_batches, total_critic_loss / num_batches, total_actor_loss / num_batches


@torch.no_grad()
def evaluate_policy_online(actor_model, embed_model, normalizer, obs_dim, action_dim, device, eval_episodes=10):
    print(f"\n--- 启动在线策略评估 (History IQL, 运行 {eval_episodes} 轮) ---")
    try:
        eval_env = gym.make("Pipeline")
    except Exception as e:
        print(f"❌ 评估错误: 无法创建 'Pipeline' 环境: {e}")
        return -float('inf')

    actor_model.eval()
    embed_model.eval()

    total_rewards = []
    zero_obs = normalizer.normalize(np.zeros(obs_dim, dtype=np.float32))

    for i in range(eval_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        terminated, truncated = False, False

        obs_normalized = normalizer.normalize(obs.reshape(-1).astype(np.float32))
        obs_history = [zero_obs] * (HIST_LEN - 1) + [obs_normalized]

        while not (terminated or truncated):
            obs_seq = np.stack(obs_history)
            obs_seq_tensor = torch.FloatTensor(obs_seq).to(device).unsqueeze(0)

            current_embed = embed_model(obs_seq_tensor)
            action_tensor = actor_model(current_embed)

            action = action_tensor.detach().cpu().numpy().flatten()
            action_clipped = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, _ = eval_env.step(action_clipped)
            next_obs_normalized = normalizer.normalize(next_obs.reshape(-1).astype(np.float32))

            obs_history.pop(0)
            obs_history.append(next_obs_normalized)

            episode_reward += reward
            obs = next_obs

        total_rewards.append(episode_reward)
        print(f"  Eval Episode {i + 1}/{eval_episodes}, Reward: {episode_reward:.2f}")

    eval_env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print("--------------------------------------------------")
    print(f"✅ 在线评估完成: 平均累积奖励: {avg_reward:.2f} +/- {std_reward:.2f}")
    print("--------------------------------------------------")
    return avg_reward


def run_iql_training(data: dict, device: torch.device):
    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    normalizer = Normalizer(data)

    dataset = SequenceRLDataset(data, hist_len=HIST_LEN, obs_dim=obs_dim, action_dim=action_dim, normalizer=normalizer)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"数据集划分: {train_size} 训练 / {val_size} 验证")

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. IQL 超参数
    iql_tau = 0.8
    iql_beta = 1.0
    discount = 0.99
    target_update_rate = 0.002
    epochs = 100
    learning_rate = 5e-5

    # 4. 初始化模型
    # 【修改】初始化时传入新增的 EMBED_DROPOUT_P 参数
    embed_net = SequenceEmbedding(
        obs_dim, HIST_LEN, EMBED_DIM, NHEAD, NUM_LAYERS, dropout_p=EMBED_DROPOUT_P
    ).to(device)
    actor = Actor(EMBED_DIM, action_dim).to(device)
    vf = ValueFunction(EMBED_DIM).to(device)

    print(f"--- 初始化 {N_CRITICS} 个 Critic 网络 ---")
    critics = nn.ModuleList([Critic(EMBED_DIM, action_dim).to(device) for _ in range(N_CRITICS)])
    target_critics = nn.ModuleList([copy.deepcopy(critic).to(device) for critic in critics])

    # 5. 优化器初始化
    actor_optimizer = optim.Adam(list(actor.parameters()) + list(embed_net.parameters()), lr=learning_rate)
    critic_parameters = []
    for critic in critics:
        critic_parameters.extend(list(critic.parameters()))
    critic_optimizer = optim.Adam(critic_parameters, lr=learning_rate)
    vf_optimizer = optim.Adam(vf.parameters(), lr=learning_rate)

    # 6. 模型和优化器加载逻辑
    os.makedirs(os.path.dirname(IQL_ACTOR_PATH), exist_ok=True)
    model_loaded = False
    if LOAD_MODELS:
        try:
            load_kwargs = {'map_location': device}
            actor.load_state_dict(torch.load(IQL_ACTOR_PATH, **load_kwargs))
            vf.load_state_dict(torch.load(IQL_VF_PATH, **load_kwargs))
            embed_net.load_state_dict(torch.load(IQL_EMBED_PATH, **load_kwargs))

            for i, critic in enumerate(critics):
                critic_path = IQL_CRITIC_PATH.replace(".pth", f"_{i + 1}.pth")
                critic.load_state_dict(torch.load(critic_path, **load_kwargs))

            model_loaded = True
            print(f"✅ 成功加载预训练 IQL 模型: Actor/VF/Embed 和 {N_CRITICS} 个 Critics")

            for i in range(N_CRITICS):
                target_critics[i].load_state_dict(critics[i].state_dict())

        except FileNotFoundError:
            print("⚠️ 未找到预训练模型文件，将从头开始训练。")
            model_loaded = False
        except Exception as e:
            print(f"❌ 加载模型时发生错误: {e}")
            print("将从头开始训练。")
            model_loaded = False

        if model_loaded:
            try:
                actor_optimizer.load_state_dict(torch.load(IQL_ACTOR_OPTIMIZER_PATH, map_location=device))
                critic_optimizer.load_state_dict(torch.load(IQL_CRITIC_OPTIMIZER_PATH, map_location=device))
                vf_optimizer.load_state_dict(torch.load(IQL_VF_OPTIMIZER_PATH, map_location=device))
                print("✅ 成功加载所有优化器状态，继续训练。")
            except Exception as e:
                print(f"⚠️ 未能加载优化器状态 ({e})，将使用加载的模型和新的优化器状态继续训练。")

    models = (actor, critics, vf, target_critics)
    optimizers = (actor_optimizer, critic_optimizer, vf_optimizer)

    # 7. 开始训练
    print(f"--- 开始 History-Aware IQL 训练 (Obs Dim={obs_dim}, Act Dim={action_dim}, Embed Dim={EMBED_DIM}) ---")
    print(f"--- 使用 {N_CRITICS} 个 Critics 和 alpha-quantile={ALPHA_QUANTILE} ---")
    print("\n--- 初始评估 (已加载/新建的模型) ---")

    current_avg_reward = evaluate_policy_online(actor, embed_net, normalizer, obs_dim, action_dim, device,
                                                eval_episodes=EVAL_EPISODES)
    best_eval_reward = current_avg_reward
    best_train_actor_loss = float('inf')

    for epoch in range(1, epochs + 1):
        vf_loss, critic_loss, actor_loss = train_one_epoch_iql(
            models, optimizers, train_loader, device, iql_tau, iql_beta, discount, embed_net,
            alpha_quantile=ALPHA_QUANTILE
        )

        for i in range(N_CRITICS):
            for param, target_param in zip(critics[i].parameters(), target_critics[i].parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)

        if actor_loss < best_train_actor_loss:
            best_train_actor_loss = actor_loss
            torch.save(actor.state_dict(), IQL_ACTOR_PATH)
            for i, critic in enumerate(critics):
                torch.save(critic.state_dict(), IQL_CRITIC_PATH.replace(".pth", f"_{i + 1}.pth"))
            torch.save(vf.state_dict(), IQL_VF_PATH)
            torch.save(embed_net.state_dict(), IQL_EMBED_PATH)
            torch.save(actor_optimizer.state_dict(), IQL_ACTOR_OPTIMIZER_PATH)
            torch.save(critic_optimizer.state_dict(), IQL_CRITIC_OPTIMIZER_PATH)
            torch.save(vf_optimizer.state_dict(), IQL_VF_OPTIMIZER_PATH)
            print(
                f"  -> Epoch {epoch:03d} New best model and OPTIMIZER STATE saved based on LOSS ({best_train_actor_loss:.6f})")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | VF Loss: {vf_loss:.6f} | Critic Loss: {critic_loss:.6f} | Actor Loss: {actor_loss:.6f}")

        if epoch % 20 == 0 or epoch == epochs:
            current_avg_reward = evaluate_policy_online(actor, embed_net, normalizer, obs_dim, action_dim, device,
                                                        eval_episodes=EVAL_EPISODES)
            if current_avg_reward > best_eval_reward:
                best_eval_reward = current_avg_reward

    print(f"\n--- 训练结束，最佳评估平均奖励: {best_eval_reward:.2f} ---")


# ---------------------------------
# --- Main Execution Block ---
# ---------------------------------
if __name__ == '__main__':
    print(f"--- 正在初始化离线数据集 for History-Aware IQL ---")
    train_data = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        env = neorl2.make("Pipeline")
        dataset_raw = env.get_dataset()
        if isinstance(dataset_raw, tuple) and len(dataset_raw) > 0:
            train_data = dataset_raw[0]
        elif isinstance(dataset_raw, dict):
            train_data = dataset_raw
        else:
            raise TypeError(f"Expected dict or tuple of dicts from get_dataset(), but received {type(dataset_raw)}")
        env.close()
    except Exception as e:
        print(f"❌ 错误: 加载 neorl2 数据集失败。请检查您的 neorl2 环境和许可证。")
        print(f"根错误: {e}")
        sys.exit(1)

    if train_data is None:
        print("❌ 错误: 数据集加载失败，程序将终止。")
        sys.exit(1)

    run_iql_training(train_data, device)
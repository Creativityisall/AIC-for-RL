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
import time

# ---------------------------------
# --- Global Parameters (来自 main.py) ---
# ---------------------------------
HIST_LEN = 30
EMBED_DIM = 256
NHEAD = 4
NUM_LAYERS = 4
MLP_HIDDEN = 256

# --- Training Hyperparameters ---
EVAL_EPISODES = 10
N_CRITICS = 6  # N 个 Critic 网络
ALPHA_QUANTILE = 0.25  # 用于 Q 值预测的 alpha 分位数
EMBED_DROPOUT_P = 0.2

# --- Diffusion 模型参数 ---
BETA_SCHEDULE = 'cosine'
N_TIMESTEPS = 20
NOISE_RATIO = 1.0
BEHAVIOR_SAMPLE = 4
EVAL_SAMPLE = 32
DETERMINISTIC_EVAL = False

# --- 文件路径 (IDQL) ---
# 注意：Critic 路径在训练时会加上 _1, _2 等后缀
IQL_ACTOR_PATH = "agent/idql_actor_model.pth"
IQL_CRITIC_PATH = "agent/idql_critic_model.pth"
IQL_VF_PATH = "agent/idql_vf_model.pth"
IQL_EMBED_PATH = "agent/idql_embed_model.pth"
IQL_ACTOR_OPTIMIZER_PATH = "agent/idql_actor_optimizer.pth"
IQL_CRITIC_OPTIMIZER_PATH = "agent/idql_critic_optimizer.pth"
IQL_VF_OPTIMIZER_PATH = "agent/idql_vf_optimizer.pth"

LOAD_MODELS = True

# --- 核心维度 ---
OBS_DIM = 52
ACTION_DIM = 1


# ----------------------------------------------------------------------
# --- START OF FILE: helpers.py ---
# ----------------------------------------------------------------------

def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape, device=t.device), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


# --------------------------------------------------------------------
# --- START OF FILE: model.py (Diffusion Model's Backbone) ---
# --------------------------------------------------------------------

class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, time_dim=32):
        super(DiffusionModel, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )
        input_dim = state_dim + action_dim + time_dim
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                   nn.Mish(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.Mish(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.Mish(),
                                   nn.Linear(hidden_size, action_dim))
        self.apply(init_weights)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)
        return out


# ------------------------------------------------
# --- START OF FILE: diffusion.py (Actor) ---
# ------------------------------------------------

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, noise_ratio,
                 beta_schedule='vp', n_timesteps=1000,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 behavior_sample=16, eval_sample=512, deterministic=False):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DiffusionModel(state_dim, action_dim)

        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio

        self.behavior_sample = behavior_sample
        self.eval_sample = eval_sample
        self.deterministic = deterministic

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 使用新的 helpers 中的 Losses
        self.loss_fn = Losses[loss_type]()

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio

    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        return x

    @torch.no_grad()
    def sample(self, state, eval=False, q_func=None, normal=False):
        if self.deterministic:
            self.noise_ratio = 0 if eval else self.max_noise_ratio
        else:
            self.noise_ratio = self.max_noise_ratio

        if normal:
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            return action

        # --- IDQL Multi-Sample Selection ---
        sample_count = self.eval_sample if eval else self.behavior_sample
        raw_batch_size = state.shape[0]
        state_rpt = state.repeat(sample_count, 1)
        shape = (state_rpt.shape[0], self.action_dim)
        action = self.p_sample_loop(state_rpt, shape)
        action.clamp_(-1., 1.)

        # q_func 应该能处理 (state_rpt, action) 形状的数据
        q1, q2 = q_func(state_rpt, action)
        q = torch.min(q1, q2)

        # 选出 Q 值最大的动作
        action = action.view(sample_count, raw_batch_size, -1).transpose(0, 1)
        q = q.view(sample_count, raw_batch_size, -1).transpose(0, 1)
        action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.action_dim)
        return action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # Advantage-Weighted Denoising Loss
        return self.p_losses(x, state, t, weights)

    def forward(self, state, eval=False, q_func=None, normal=False):
        return self.sample(state, eval, q_func, normal)


# ----------------------------------------------------
# --- Sequence Embedding and IQL/IDQL Components ---
# ----------------------------------------------------

class Normalizer:
    def __init__(self, data: dict):
        all_obs = np.concatenate([data['obs'], data['next_obs']], axis=0).astype(np.float32)
        self.mean = np.mean(all_obs, axis=0)
        self.std = np.std(all_obs, axis=0)
        self.std[self.std < 1e-5] = 1e-5
        print("✅ Normalizer initialized.")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std


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
    def __init__(self, obs_dim, hist_len, embed_dim, nhead, num_layers, dropout_p: float):
        super().__init__()
        self.input_embed = nn.Linear(obs_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=hist_len)
        self.embed_dropout = nn.Dropout(p=dropout_p)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=MLP_HIDDEN,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        print(f"[SequenceEmbedding] Initialized: Obs({obs_dim}) -> Embed({embed_dim})")

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_embed(obs_seq)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)
        transformer_out = self.transformer_encoder(x)
        # 取最后一个时间步的输出作为状态嵌入
        return transformer_out[:, -1, :]


def build_mlp(input_dim, output_dim, hidden_units=[256, 256]):
    layers = [nn.Linear(input_dim, hidden_units[0]), nn.ReLU()]
    for i in range(len(hidden_units) - 1):
        layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_units[-1], output_dim))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    # 这是 Q 网络 $Q(s, a)$
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.net = build_mlp(embed_dim + action_dim, 1)

    def forward(self, embed, action):
        return self.net(torch.cat([embed, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    # 这是价值网络 $V(s)$
    def __init__(self, embed_dim):
        super().__init__()
        self.net = build_mlp(embed_dim, 1)

    def forward(self, embed):
        return self.net(embed).squeeze(-1)


class SequenceRLDataset(Dataset):
    def __init__(self, data: dict, hist_len: int, obs_dim: int, action_dim: int,
                 normalizer: Normalizer):
        self.obs = normalizer.normalize(data['obs'].astype(np.float32))
        self.next_obs = normalizer.normalize(data['next_obs'].astype(np.float32))
        self.act = data['action'].astype(np.float32)
        self.rew = data['reward'].astype(np.float32).squeeze()
        self.done = data['done'].astype(np.float32).squeeze()
        self.hist_len = hist_len
        self.total_size = len(self.obs)
        self.start_idx = hist_len - 1
        self.available_samples = self.total_size - self.start_idx - 1

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


def expectile_loss(diff, expectile=0.7):
    """用于 Value Function 的 Expectile Loss (IQL)"""
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return (weight * (diff ** 2)).mean()


def huber_loss(diff, delta: float = 1.0):
    """用于 Critic 的 Huber Loss (来自 IQL 论文的变体)"""
    abs_diff = torch.abs(diff)
    # L_H^δ(x) = { 1/(2δ) * x^2     if |x| <= δ
    #           { |x| - 1/2 * δ    if |x| > δ
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
            # 计算所有 N 个 target critics 的 alpha 分位数作为 Q target for VF and Actor
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
        delta_for_huber = 1.0  # Huber Loss 的参数
        # 迭代所有 N 个 critics，并使用 Huber Loss 聚合损失
        for critic in critics:
            q_pred = critic(current_embed.detach(), act)
            critic_loss += huber_loss(q_pred - q_target_for_critic, delta=delta_for_huber)

        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_optimizer.step()
        total_critic_loss += critic_loss.item()

        # --- 3. 训练策略网络 (Diffusion Actor) ---
        with torch.no_grad():
            advantage = q_target_for_vf_and_actor - vf_pred.detach()
            exp_advantage = torch.exp(iql_beta * advantage).clamp(max=100.0)

        # 使用 Diffusion Actor 的 Advantage-Weighted Denoising Loss
        actor_loss = actor.loss(act, current_embed, weights=exp_advantage)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        total_actor_loss += actor_loss.item()

    num_batches = len(loader)
    return total_vf_loss / num_batches, total_critic_loss / num_batches, total_actor_loss / num_batches


@torch.no_grad()
def evaluate_policy_online(actor_model, embed_model, critics, normalizer, obs_dim, action_dim, device,
                           eval_episodes=10):
    print(f"\n--- 启动在线策略评估 (IDQL, 运行 {eval_episodes} 轮) ---")
    try:
        eval_env = gym.make("Pipeline")
    except Exception as e:
        print(f"❌ 评估错误: 无法创建 'Pipeline' 环境: {e}")
        return -float('inf')

    actor_model.eval()
    embed_model.eval()
    for critic in critics:
        critic.eval()

    total_rewards = []
    zero_obs = normalizer.normalize(np.zeros(obs_dim, dtype=np.float32))

    # 评估时使用前两个 Critic 进行 $\min(Q_1, Q_2)$ 采样
    q_func_for_eval = lambda s, a: (critics[0](s, a), critics[1](s, a))

    for i in range(eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        terminated, truncated = False, False
        obs_normalized = normalizer.normalize(obs.reshape(-1).astype(np.float32))
        obs_history = [zero_obs] * (HIST_LEN - 1) + [obs_normalized]

        while not (terminated or truncated):
            obs_seq = np.stack(obs_history)
            obs_seq_tensor = torch.FloatTensor(obs_seq).to(device).unsqueeze(0)
            current_embed = embed_model(obs_seq_tensor)

            # 使用 Diffusion Actor 的多样本采样策略
            action_tensor = actor_model(current_embed, eval=True, q_func=q_func_for_eval, normal=False)
            action = action_tensor.detach().cpu().numpy().flatten()
            action_clipped = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, _ = eval_env.step(action_clipped)
            next_obs_normalized = normalizer.normalize(next_obs.reshape(-1).astype(np.float32))
            obs_history.pop(0)
            obs_history.append(next_obs_normalized)
            episode_reward += reward

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
    train_ds, _ = random_split(dataset, [train_size, val_size])

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. IDQL 超参数
    iql_tau = 0.8
    iql_beta = 1.0
    discount = 0.99
    target_update_rate = 0.002
    epochs = 2000
    learning_rate = 5e-5

    # 4. 初始化模型 (Actor 是 Diffusion Model)
    embed_net = SequenceEmbedding(obs_dim, HIST_LEN, EMBED_DIM, NHEAD, NUM_LAYERS, dropout_p=EMBED_DROPOUT_P).to(device)
    actor = Diffusion(
        state_dim=EMBED_DIM, action_dim=action_dim, noise_ratio=NOISE_RATIO,
        beta_schedule=BETA_SCHEDULE, n_timesteps=N_TIMESTEPS, behavior_sample=BEHAVIOR_SAMPLE,
        eval_sample=EVAL_SAMPLE, deterministic=DETERMINISTIC_EVAL
    ).to(device)
    vf = ValueFunction(EMBED_DIM).to(device)

    # 初始化 N 个 critic 和 target critic
    print(f"--- 初始化 {N_CRITICS} 个 Critic 网络 ---")
    critics = nn.ModuleList([Critic(EMBED_DIM, action_dim).to(device) for _ in range(N_CRITICS)])
    target_critics = nn.ModuleList([copy.deepcopy(critic).to(device) for critic in critics])

    # 5. 优化器初始化
    # Actor/Embed Optimizer
    actor_optimizer = optim.Adam(list(actor.parameters()) + list(embed_net.parameters()), lr=learning_rate)
    # Critic Optimizer (包含所有 N 个 Critics)
    critic_parameters = []
    for critic in critics:
        critic_parameters.extend(list(critic.parameters()))
    critic_optimizer = optim.Adam(critic_parameters, lr=learning_rate)
    # VF Optimizer
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
            print(f"✅ 成功加载预训练 IDQL 模型: Actor/VF/Embed 和 {N_CRITICS} 个 Critics")

            for i in range(N_CRITICS):
                target_critics[i].load_state_dict(critics[i].state_dict())

        except FileNotFoundError:
            print("⚠️ 未找到预训练模型文件，将从头开始训练。")
        except Exception as e:
            print(f"❌ 加载模型时发生错误: {e}")
            print("将从头开始训练。")

        if model_loaded:
            try:
                actor_optimizer.load_state_dict(torch.load(IQL_ACTOR_OPTIMIZER_PATH, map_location=device))
                critic_optimizer.load_state_dict(torch.load(IQL_CRITIC_OPTIMIZER_PATH, map_location=device))
                vf_optimizer.load_state_dict(torch.load(IQL_VF_OPTIMIZER_PATH, map_location=device))
                print("✅ 成功加载所有优化器状态，继续训练。")
            except Exception:
                print(f"⚠️ 未能加载优化器状态，将使用加载的模型和新的优化器状态继续训练。")

    models = (actor, critics, vf, target_critics)
    optimizers = (actor_optimizer, critic_optimizer, vf_optimizer)

    # 7. 开始训练
    print(f"--- 开始 History-Aware IDQL 训练 (Obs Dim={obs_dim}, Act Dim={action_dim}, Embed Dim={EMBED_DIM}) ---")
    print(f"--- 使用 {N_CRITICS} 个 Critics 和 alpha-quantile={ALPHA_QUANTILE} ---")

    current_avg_reward = evaluate_policy_online(actor, embed_net, critics, normalizer, obs_dim, action_dim, device,
                                                eval_episodes=EVAL_EPISODES)
    best_eval_reward = current_avg_reward
    best_train_actor_loss = float('inf')

    for epoch in range(1, epochs + 1):
        vf_loss, critic_loss, actor_loss = train_one_epoch_iql(
            models, optimizers, train_loader, device, iql_tau, iql_beta, discount, embed_net,
            alpha_quantile=ALPHA_QUANTILE
        )

        # 软更新所有 N 个 target critics
        for i in range(N_CRITICS):
            for param, target_param in zip(critics[i].parameters(), target_critics[i].parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)

        if actor_loss < best_train_actor_loss:
            best_train_actor_loss = actor_loss
            # Save all components
            torch.save(actor.state_dict(), IQL_ACTOR_PATH)
            for i, critic in enumerate(critics):
                torch.save(critic.state_dict(), IQL_CRITIC_PATH.replace(".pth", f"_{i + 1}.pth"))
            torch.save(vf.state_dict(), IQL_VF_PATH)
            torch.save(embed_net.state_dict(), IQL_EMBED_PATH)
            torch.save(actor_optimizer.state_dict(), IQL_ACTOR_OPTIMIZER_PATH)
            torch.save(critic_optimizer.state_dict(), IQL_CRITIC_OPTIMIZER_PATH)
            torch.save(vf_optimizer.state_dict(), IQL_VF_OPTIMIZER_PATH)
            print(
                f"  -> Epoch {epoch:03d} New best model and OPTIMIZER STATE saved based on LOSS ({best_train_actor_loss:.4f})")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | VF Loss: {vf_loss:.4f} | Critic Loss: {critic_loss:.4f} | Actor Loss: {actor_loss:.4f}")

        if epoch % 50 == 0 or epoch == epochs:
            current_avg_reward = evaluate_policy_online(actor, embed_net, critics, normalizer, obs_dim, action_dim,
                                                        device,
                                                        eval_episodes=EVAL_EPISODES)
            if current_avg_reward > best_eval_reward:
                best_eval_reward = current_avg_reward

    print(f"\n--- 训练结束，最佳评估平均奖励: {best_eval_reward:.2f} ---")


# ---------------------------------
# --- Main Execution Block ---
# ---------------------------------
if __name__ == '__main__':
    print(f"--- 正在初始化离线数据集 for History-Aware IDQL ---")
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
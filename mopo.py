import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import numpy as np
import copy
import os
import gymnasium as gym
import neorl2  # ensure installed

# ================== 全局设置 ==================
OBS_DIM = 52
HISTORY_LEN = 1
STATE_DIM = OBS_DIM * HISTORY_LEN
ACTION_DIM = 1

HIDDEN_DIM = 256
NUM_ENSEMBLE = 7
ROLLOUT_LENGTH = 3
UNCERTAINTY_LAMBDA = 7.0

BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR_A = 1e-5
LR_C = 1e-5
LR_DYN = 3e-4
ALPHA = 0.2
TARGET_ENTROPY = -float(ACTION_DIM)
POLICY_UPDATE_FREQ = 2

# 为了 demo 快速运行，这里可以把下面几个数值调小
DYNAMICS_TRAIN_EPOCHS = 3000  # 原来 30000，测试用可减小
TRAIN_EPOCHES = 50            # 总轮数（外循环）
STEPS_EACH_EPOCH = 800        # 每轮策略训练步数
EVAL_EPISODES = 10

# === BC预热参数 (可以调小以快速测试) ===
BC_EPOCHS = 3000
BC_BATCH_SIZE = 256
BC_LR = 3e-4
BC_LOG_INTERVAL = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"[Config] State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")


# ================== Normalizer ==================
class Normalizer:
    def __init__(self, size):
        self.size = size
        self.mean = np.zeros(size)
        self.std = np.ones(size)

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-12] = 1.0
        print("[Normalizer] Fitted with data. Mean and std calculated.")

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            device = data.device
            mean = torch.FloatTensor(self.mean).to(device)
            std = torch.FloatTensor(self.std).to(device)
            return (data - mean) / std
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            device = data.device
            mean = torch.FloatTensor(self.mean).to(device)
            std = torch.FloatTensor(self.std).to(device)
            return data * std + mean
        return data * self.std + self.mean


# ================== Replay Buffer ==================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.device = DEVICE

    def add(self, s, a, s_next, r, done):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_next
        self.reward[self.ptr] = r
        self.not_done[self.ptr] = 1. - float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty.")
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


# ================== 数据加载 ==================
def load_neorl2_dataset(replay_buffer: ReplayBuffer):
    print("Loading NeoRL-2 'Pipeline' dataset...")
    env = gym.make("Pipeline")
    train_data, _ = env.get_dataset()
    env.close()

    obs = train_data["obs"]
    actions = train_data["action"]
    next_obs = train_data["next_obs"]
    rewards = train_data["reward"]
    terminals = np.logical_or(train_data["done"], train_data["truncated"])

    num_transitions = len(obs)
    for i in range(num_transitions):
        # ensure shapes: obs[i] shape (STATE_DIM,), actions[i] shape (ACTION_DIM,)
        replay_buffer.add(
            obs[i].astype(np.float32),
            actions[i].astype(np.float32),
            next_obs[i].astype(np.float32),
            np.array([rewards[i]], dtype=np.float32),
            bool(terminals[i])
        )

    print(f"✅ Loaded {num_transitions} transitions into buffer.")


# ================== 动态模型 ==================
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        output_dim = (state_dim + 1) * 2
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.max_log_std = nn.Parameter(torch.ones(state_dim + 1) * 0.5, requires_grad=False)
        self.min_log_std = nn.Parameter(torch.ones(state_dim + 1) * -10.0, requires_grad=False)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        output = self.mlp(sa)
        delta_s_mean, delta_s_log_std, r_mean, r_log_std = torch.split(
            output, [STATE_DIM, STATE_DIM, 1, 1], dim=-1
        )
        mean = torch.cat([delta_s_mean, r_mean], dim=-1)
        log_std = torch.cat([delta_s_log_std, r_log_std], dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std


class EnsembleDynamicsModel:
    def __init__(self, state_dim, action_dim, normalizer, num_ensemble=NUM_ENSEMBLE):
        self.num_ensemble = num_ensemble
        self.models = [DynamicsModel(state_dim, action_dim).to(DEVICE) for _ in range(num_ensemble)]
        self.optimizers = [optim.Adam(m.parameters(), lr=LR_DYN) for m in self.models]
        self.normalizer = normalizer
        self.state_dim = state_dim

    def train(self, real_buffer, epochs=DYNAMICS_TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        if real_buffer.size < batch_size:
            raise RuntimeError("Not enough data in real buffer to train dynamics.")
        for epoch in range(epochs):
            for i in range(self.num_ensemble):
                m = self.models[i]
                opt = self.optimizers[i]
                s, a, s_next, r, _ = real_buffer.sample(batch_size)
                s_norm = self.normalizer.transform(s)
                s_next_norm = self.normalizer.transform(s_next)
                delta_s = s_next_norm - s_norm
                target = torch.cat([delta_s, r], dim=-1)
                mean, log_std = m(s_norm, a)
                std = torch.exp(log_std)
                # gaussian_nll_loss expects input = mean, target, var? use F.gaussian_nll_loss(mean, target, var) where var=std**2
                var = std * std
                loss = F.gaussian_nll_loss(target, mean, var, reduction='mean')
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=40.0) # Dynamics Model Gradient Clipping
                opt.step()
            if (epoch + 1) % 10 == 0:
                print(f"[Dynamics Train] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    def step(self, s, a):
        with torch.no_grad():
            means = []
            log_stds = []
            s_norm = self.normalizer.transform(s)
            for m in self.models:
                mean, log_std = m(s_norm, a)
                means.append(mean)
                log_stds.append(log_std)
            means = torch.stack(means)  # (num_ens, B, dim)
            log_stds = torch.stack(log_stds)
            stds = torch.exp(log_stds)
            samples = means + torch.randn_like(means) * stds
            idx = torch.randint(0, self.num_ensemble, (s.size(0),), device=DEVICE)
            batch_idx = torch.arange(0, s.size(0), device=DEVICE)
            sample = samples[idx, batch_idx]  # (B, dim)
            delta_s, r = torch.split(sample, [self.state_dim, 1], dim=-1)
            s_next_norm = s_norm + delta_s
            s_next = self.normalizer.inverse_transform(s_next_norm)

            # 不确定性估计：对 ensemble 的 delta_mean 做 std，然后算范数（每个维度 std，然后 L2）
            delta_means = means[:, :, :self.state_dim]  # (num_ens, B, state_dim)
            # per-dimension std over ensemble
            per_dim_std = delta_means.std(dim=0)  # (B, state_dim)
            # combine dims (L2)
            uncertainty = torch.norm(per_dim_std, dim=-1, keepdim=True)  # (B,1)

            # reward 平均
            reward_means = means[:, :, self.state_dim:]  # (num_ens, B, 1)
            mean_r = reward_means.mean(dim=0)  # (B,1)
            penalized_r = mean_r - UNCERTAINTY_LAMBDA * uncertainty  # (B,1)

            # safety checks
            if torch.isnan(s_next).any() or torch.isinf(s_next).any():
                invalid = torch.logical_or(torch.isnan(s_next).any(dim=1), torch.isinf(s_next).any(dim=1))
                invalid = invalid.to(device=s_next.device)
                s_next[invalid] = s[invalid]
                penalized_r[invalid] = -10.0

            return s_next, penalized_r


# ================== 策略网络 (SAC) ==================
class TanhNormal(TransformedDistribution):
    def __init__(self, base_distribution, transforms):
        super().__init__(base_distribution, transforms)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):
        x = self.mlp(s)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp()
        base = Normal(mean, std)
        dist = TanhNormal(base, [TanhTransform(cache_size=1)])
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
        return a, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(self, state_dim, action_dim, normalizer):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_targ = copy.deepcopy(self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_C)
        self.log_alpha = torch.tensor(np.log(ALPHA), device=DEVICE, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR_A)
        self.target_entropy = TARGET_ENTROPY
        self.normalizer = normalizer
        self.total_it = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, s, deterministic=False):
        # accept numpy array or torch tensor
        was_tensor = isinstance(s, torch.Tensor)
        if was_tensor:
            s_np = s.cpu().numpy()
        else:
            s_np = np.array(s)
        single = False
        if s_np.ndim == 1:
            s_np = s_np.reshape(1, -1)
            single = True
        s_norm = self.normalizer.transform(s_np)
        s_t = torch.FloatTensor(s_norm).to(DEVICE)
        with torch.no_grad():
            x = self.actor.mlp(s_t)
            mean = self.actor.mean_layer(x)
            if deterministic:
                a = torch.tanh(mean)
            else:
                log_std = self.actor.log_std_layer(x).clamp(-20, 2)
                std = log_std.exp()
                dist = TanhNormal(Normal(mean, std), [TanhTransform(cache_size=1)])
                a = dist.rsample()
        a_np = a.cpu().numpy()
        if single:
            return a_np.flatten()
        return a_np

    # ========== BC 预热阶段 ==========
    def behavior_cloning_train(self, buffer, epochs=BC_EPOCHS, batch_size=BC_BATCH_SIZE):
        print(f"--- 行为克隆 (BC) 预训练开始，共 {epochs} steps ---")
        opt = optim.Adam(self.actor.parameters(), lr=BC_LR)
        for it in range(epochs):
            s, a, _, _, _ = buffer.sample(batch_size)
            s_norm = self.normalizer.transform(s)
            pred_a, _, _ = self.actor(s_norm)
            loss = F.mse_loss(pred_a, a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # BC Gradient Clipping
            opt.step()
            if (it + 1) % BC_LOG_INTERVAL == 0:
                print(f"[BC] Step {it + 1}/{epochs}, Loss={loss.item():.6f}")
        print("--- 行为克隆预热结束 ---")

    def train(self, batch):
        self.total_it += 1
        s, a, s_next, r, not_done = batch
        s_norm = self.normalizer.transform(s)
        s_next_norm = self.normalizer.transform(s_next)

        with torch.no_grad():
            a_next, logp_next, _ = self.actor(s_next_norm)
            q1_t, q2_t = self.critic_targ(s_next_norm, a_next)
            q_t = torch.min(q1_t, q2_t)
            y = r + not_done * GAMMA * (q_t - self.alpha * logp_next)

        q1, q2 = self.critic(s_norm, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # --- Critic 梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        if self.total_it % POLICY_UPDATE_FREQ == 0:
            a_new, logp_new, _ = self.actor(s_norm)
            q1_new, q2_new = self.critic(s_norm, a_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * logp_new - q_new).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            # --- Actor 梯度裁剪 ---
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            # --- Alpha 梯度裁剪 ---
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_opt.step()

            # soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_targ.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# ================== MOPO 主体 ==================
class MOPO:
    def __init__(self, state_dim, action_dim):
        self.real_buffer = ReplayBuffer(state_dim, action_dim)
        load_neorl2_dataset(self.real_buffer)
        if self.real_buffer.size == 0:
            raise RuntimeError("Real buffer empty after loading dataset.")
        self.state_norm = Normalizer(state_dim)
        self.state_norm.fit(self.real_buffer.state[:self.real_buffer.size])
        self.dynamics = EnsembleDynamicsModel(state_dim, action_dim, self.state_norm, num_ensemble=NUM_ENSEMBLE)
        self.policy = SAC(state_dim, action_dim, self.state_norm)
        self.virtual_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(ROLLOUT_LENGTH * 1e4))

    def behavior_cloning_pretrain(self, epochs=BC_EPOCHS):
        print("\n--- 步骤 0: 行为克隆 (BC) 预热 ---")
        self.policy.behavior_cloning_train(self.real_buffer, epochs=epochs, batch_size=BC_BATCH_SIZE)
        print("--- BC 预热完毕 ---")

    def train_dynamics(self):
        print("--- 步骤 1: 开始训练动态模型 ---")
        self.dynamics.train(self.real_buffer, epochs=DYNAMICS_TRAIN_EPOCHS, batch_size=BATCH_SIZE)
        print("--- 动态模型训练完毕 ---")

    def generate_virtual_data(self):
        print("--- 步骤 2: 开始生成虚拟数据 ---")
        # reset virtual buffer
        self.virtual_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, max_size=self.virtual_buffer.max_size)
        # number of starting states = virtual_buffer.max_size // rollout_length
        num_rollouts = max(1, self.virtual_buffer.max_size // max(1, ROLLOUT_LENGTH))
        if self.real_buffer.size < num_rollouts:
            num_rollouts = self.real_buffer.size
        start_states, _, _, _, _ = self.real_buffer.sample(num_rollouts)
        states = start_states
        for _ in range(ROLLOUT_LENGTH):
            actions = self.policy.select_action(states)  # returns numpy (B,1)
            actions_t = torch.FloatTensor(actions).to(DEVICE)
            next_states, penalized_rewards = self.dynamics.step(states, actions_t)
            for i in range(states.size(0)):
                s_np = states[i].cpu().numpy()
                a_np = actions[i]
                ns_np = next_states[i].cpu().numpy()
                r_np = np.array([float(penalized_rewards[i].item())], dtype=np.float32)
                self.virtual_buffer.add(s_np, a_np, ns_np, r_np, False)
            states = next_states
        print(f"--- 虚拟数据生成完毕，大小: {self.virtual_buffer.size} ---")

    def train_policy(self, iters):
        print("--- 步骤 3: 训练策略网络 ---")
        if self.virtual_buffer.size < 1:
            print("虚拟 buffer 为空，跳过策略训练。")
            return
        for i in range(iters):
            # sample real & virtual batches (proportion 90% / 10%)
            rb = self.real_buffer.sample(BATCH_SIZE * 9 // 10)
            vb = self.virtual_buffer.sample(BATCH_SIZE * 1 // 10)
            batch = tuple(torch.cat([r, v], dim=0) for r, v in zip(rb, vb))
            self.policy.train(batch)
            if (i + 1) % max(1, (iters // 10)) == 0:
                print(f"[Policy Train] {i + 1}/{iters}")
        print("--- 策略训练阶段完成 ---")

    def evaluate_policy(self, episodes=EVAL_EPISODES):
        print(f"--- 评估策略 ({episodes} episodes) ---")
        env = gym.make("Pipeline")
        rews = []
        for i in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            while True:
                a = self.policy.select_action(obs, deterministic=True)
                next_obs, r, term, trunc, _ = env.step(a)
                ep_rew += float(r)
                obs = next_obs
                if term or trunc:
                    break
            rews.append(ep_rew)
            print(f"   Episode {i+1}, Reward={ep_rew:.2f}")
        env.close()
        print(f"✅ Eval mean reward: {np.mean(rews):.2f} ± {np.std(rews):.2f}")

    def save(self, base_path="mopo_pipeline_model"):
        os.makedirs(base_path, exist_ok=True)
        torch.save(self.policy.actor.state_dict(), os.path.join(base_path, "actor.pth"))
        torch.save(self.policy.critic.state_dict(), os.path.join(base_path, "critic.pth"))
        dyn_dir = os.path.join(base_path, "dynamics")
        os.makedirs(dyn_dir, exist_ok=True)
        for i, m in enumerate(self.dynamics.models):
            torch.save(m.state_dict(), os.path.join(dyn_dir, f"model_{i}.pth"))
        np.savez(os.path.join(base_path, "normalizer.npz"), mean=self.state_norm.mean, std=self.state_norm.std)
        print(f"Models saved to {base_path}")

    def run(self, num_epochs=TRAIN_EPOCHES, policy_train_iterations=STEPS_EACH_EPOCH, eval_freq=1):
        # 0. BC pretrain
        self.behavior_cloning_pretrain(epochs=BC_EPOCHS)
        # optional quick eval after BC
        print("\n--- 评估 BC 预热策略 ---")
        self.evaluate_policy(episodes=2)

        # 1. train dynamics
        self.train_dynamics()

        # main loop
        for epoch in range(num_epochs):
            print(f"\n=== MOPO Epoch {epoch + 1}/{num_epochs} ===")
            # 2. generate virtual data
            self.generate_virtual_data()
            # 3. train policy with combined data
            self.train_policy(policy_train_iterations)
            # 4. evaluate
            if (epoch + 1) % eval_freq == 0:
                self.evaluate_policy()

        # final save
        print("\n--- 训练结束，保存模型 ---")
        self.save()


# ================== main ==================
if __name__ == "__main__":
    # To allow quick smoke test, you may reduce epochs above.
    agent = MOPO(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    try:
        agent.run(num_epochs=TRAIN_EPOCHES, policy_train_iterations=STEPS_EACH_EPOCH, eval_freq=1)
    except KeyboardInterrupt:
        print("训练被人工中断，正在保存当前模型...")
        agent.save(base_path="mopo_interrupted")
        raise

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm

from logger_manager import logger

### Model ###
class Mlp(nn.Module):
    def __init__(self,
                 input_dim: int = 150,
                 hidden_dim: int = 256,
                 output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def train_epoch(self, batch, optimizer):
        self.train()
        obs = batch['s']
        act = batch['a']

        pred = self.forward(obs)
        loss = F.mse_loss(pred, act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class ADM(nn.Module):
    def __init__(self, state_dim, action_dim, max_backtrack_len=5, hidden_dim=256):
        super(ADM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_backtrack_len = max_backtrack_len

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,  # 状态嵌入 + 动作嵌入
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,  # 输入格式：(batch, seq_len, hidden_dim)
            dropout=0.1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),  # 增加中间层维度
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (state_dim + 1))
        )
    
    def forward(self, state, action_seq):
        batch_size, k, _ = action_seq.shape
        state_emb = self.state_proj(state.unsqueeze(1).repeat(1, k, 1))
        action_emb = self.action_proj(action_seq)
        
        gru_input = torch.cat([state_emb, action_emb], dim=-1)
        gru_out, _ = self.gru(gru_input)
        gru_final = gru_out[:, -1, :]
        
        out = self.output_proj(gru_final)

        state_mean = out[:, :self.state_dim]
        state_log_std = torch.tanh(out[:, self.state_dim:2 * self.state_dim])  # 限制范围
        state_std = torch.exp(state_log_std) * 0.5 + 0.5
        
        reward_mean = out[:, 2 * self.state_dim:2 * self.state_dim + 1]

        reward_log_std = torch.tanh(out[:, 2 * self.state_dim + 1:])
        reward_std = torch.exp(reward_log_std) * 0.5 + 0.5

        return state_mean, state_std, reward_mean, reward_std
    
    def sample_next(self, state, action_seq):
        """采样目标状态和奖励（论文中用高斯采样）"""
        state_mean, state_std, reward_mean, reward_std = self.forward(state, action_seq)
        # 高斯分布采样
        state_dist = Normal(state_mean, state_std)
        reward_dist = Normal(reward_mean, reward_std)
        next_state = state_dist.rsample()  # 重参数化采样（便于梯度回传）
        reward = reward_dist.rsample()
        return next_state, reward
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def reset(self, *args, **kwargs):
        dummy_obs = np.random.rand(self.state_dim).astype(np.float32)
        return dummy_obs
    
    @torch.no_grad()
    def step(self, state, action):
        state = to_torch(state)
        action = to_torch(action)

        next_state, reward = self.sample_next(state.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))
        done = torch.zeros((state.shape[0], 1))  # 简化处理，假设不终止

        next_state = next_state.squeeze(0).cpu().numpy()
        reward = reward.squeeze(0).cpu().numpy()
        return next_state, reward, done, {}


### Train ###
def calculate_adm_uncertainty(adm, state, action_history, max_backtrack_len=5):
    batch_size = state.shape[0]
    state_preds = []  # 存储不同回溯长度的状态预测
    
    # 遍历所有可能的回溯长度k（1<=k<=m）
    for k in range(1, max_backtrack_len + 1):
        # 截取前k个动作作为动作序列（action_history是最近max_backtrack_len个动作）
        action_seq = action_history[:, :k, :]  # (batch, k, action_dim)
        # 预测目标状态（当前回溯长度k对应的目标状态）
        state_mean, _, _, _ = adm(state, action_seq)
        state_preds.append(state_mean.unsqueeze(1))  # (batch, 1, state_dim)
    
    # 拼接所有回溯长度的预测，计算方差（论文中用L1范数的期望）
    state_preds = torch.cat(state_preds, dim=1)  # (batch, m, state_dim)
    mean_pred = state_preds.mean(dim=1, keepdim=True)  # (batch, 1, state_dim)：所有k的平均
    var_per_dim = ((state_preds - mean_pred) ** 2).mean(dim=1)  # (batch, state_dim)：每维方差
    uncertainty = var_per_dim.norm(p=1, dim=1, keepdim=True)  # L1范数（论文中用L1）
    return uncertainty

class ADMPO_OFF:
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            policy,
            policy_opt,
            max_backtrack_len=5, 
            hidden_dim=256, 
            lr_adm=3e-4,
            weight_decay=1e-5,
            beta=2.5,
            T_max=1000,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_backtrack_len = max_backtrack_len
        self.beta = beta
        self.loss_fn = nn.MSELoss()

        self.adm = ADM(state_dim, action_dim, max_backtrack_len, hidden_dim).to(device)
        self.policy = policy.to(device) if policy is not None else nn.Linear(state_dim, action_dim).to(device)

        self.adm_opt = optim.Adam(self.adm.parameters(), lr=lr_adm, weight_decay=weight_decay)
        self.policy_opt = policy_opt if policy_opt is not None else optim.Adam(self.policy.parameters(), lr=lr_adm)

        self.adm_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.adm_opt, T_max=T_max)
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.policy_opt, T_max=T_max)

    def train_adm(self, replay_buffer, batch_size=256, epochs=10, std_penalty_coef=0.01):
        for idx in tqdm(range(epochs), desc="Training ADM"):
            # 采样批量数据：(s_t, a_t:t+k-1, s_t+k, r_t+k)，k为随机回溯长度
            batch = replay_buffer.sample(batch_size, self.max_backtrack_len)
            s_t = batch['s_t']  # (batch, state_dim)
            action_seq = batch['action_seq']  # (batch, k, action_dim)，k随机1~m
            s_tk = batch['s_tk']  # (batch, state_dim)：目标状态
            r_tk = batch['r_tk']  # (batch, 1)：目标奖励
            
            # ADM预测
            s_mean, s_std, r_mean, r_std = self.adm(s_t, action_seq)

            # 添加数值稳定性
            s_std = torch.clamp(s_std, min=1e-4)
            r_std = torch.clamp(r_std, min=1e-4)

            # 对数似然损失（高斯分布）
            s_dist = Normal(s_mean, s_std)
            r_dist = Normal(r_mean, r_std)
            loss_s = - s_dist.log_prob(s_tk).mean()  # 状态预测损失
            loss_r = - r_dist.log_prob(r_tk).mean()  # 奖励预测损失

            # 添加标准差正则化，防止过小
            std_penalty = std_penalty_coef * (torch.exp(-s_std * 10).mean() + torch.exp(-r_std * 10).mean())

            loss_adm = loss_s + loss_r + std_penalty

            # 反向传播
            self.adm_opt.zero_grad()
            loss_adm.backward()
            torch.nn.utils.clip_grad_norm_(self.adm.parameters(), max_norm=1.0)
            self.adm_opt.step()

            if idx % 500 == 0:
                logger.info(f"Epoch {idx}, ADM Loss: {loss_adm.item():.4f}, State Loss: {loss_s.item():.4f}, Reward Loss: {loss_r.item():.4f}")

    def model_rollout(self, initial_state, model_buffer, rollout_len=10):
        """简化的模型rollout实现"""
        batch_size = initial_state.shape[0]
        current_s = initial_state.to(self.device)
        
        # 初始化动作历史
        action_history = torch.zeros(batch_size, self.max_backtrack_len, self.action_dim).to(self.device)

        for step in range(rollout_len):
            # 1. 策略采样动作
            with torch.no_grad():
                if hasattr(self.policy, 'sample'):
                    action, _, _ = self.policy.sample(current_s)
                else:
                    action = self.policy(current_s)
            
            # 2. 更新动作历史
            action_history = torch.roll(action_history, shifts=-1, dims=1)
            action_history[:, -1, :] = action
            
            # 3. 随机选择回溯长度k
            k = np.random.randint(1, self.max_backtrack_len + 1)
            backtrack_action_seq = action_history[:, -k:, :]
            
            # 4. ADM预测下一个状态和奖励
            s_next, r_raw = self.adm.sample_next(current_s, backtrack_action_seq)
            
            # 5. 计算不确定性并惩罚奖励
            uncertainty = calculate_adm_uncertainty(self.adm, current_s, action_history, self.max_backtrack_len)
            r_penalized = r_raw - self.beta * uncertainty
            
            # 6. 存储到模型buffer
            for i in range(batch_size):
                model_buffer.add(
                    current_s[i].cpu().detach().numpy(),
                    action[i].cpu().detach().numpy(),
                    s_next[i].cpu().detach().numpy(),
                    r_penalized[i].cpu().detach().numpy(),
                    done = np.array([0.0])
                )
            
            current_s = s_next.detach()

    def train_policy(self, real_buffer, model_buffer, batch_size=256, epochs=5):
        total_loss = 0.0
        for _ in range(epochs):
            batch_real = real_buffer.sample(batch_size // 2)
            batch_model = model_buffer.sample(batch_size // 2)
            batch = {
                's': torch.cat([batch_real['s'], batch_model['s']], dim=0).to(self.device),
                'a': torch.cat([batch_real['a'], batch_model['a']], dim=0).to(self.device),
                'r': torch.cat([batch_real['r'], batch_model['r']], dim=0).to(self.device),
                's_next': torch.cat([batch_real['s_next'], batch_model['s_next']], dim=0).to(self.device),
                'done': torch.cat([batch_real['done'], batch_model['done']], dim=0).to(self.device)
            }
            loss = self.policy.train_epoch(batch, self.policy_opt)
            total_loss += loss

        return total_loss / epochs
    
    def evaluate_policy(self, eval_buffer, batch_size=256):
        self.policy.eval()
        total_loss = 0.0
        num_batches = eval_buffer.size // batch_size
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Evaluating Policy"):
                batch = eval_buffer.sample(batch_size)
                s = batch['s'].to(self.device)
                a = batch['a'].to(self.device)

                pred_a = self.policy(s)
                loss = self.loss_fn(pred_a, a)
                total_loss += loss.item()
        return total_loss / num_batches

    def save_adm(self, filepath):
        self.adm.save(filepath+"_adm.pth")

    def load_adm(self, filepath):
        self.adm.load(filepath+"_adm.pth")

    def save_policy(self, filepath):
        torch.save(self.policy.state_dict(), filepath+"_policy.pth")

    def load_policy(self, filepath):
        self.policy.load_state_dict(torch.load(filepath+"_policy.pth"), self.device)

### Buffer ###
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # 存储单步数据（用于策略训练）
        self.s = torch.zeros((max_size, state_dim))
        self.a = torch.zeros((max_size, action_dim))
        self.r = torch.zeros((max_size, 1))
        self.s_next = torch.zeros((max_size, state_dim))
        self.done = torch.zeros((max_size, 1))
    
    def add(self, s, a, s_next, r, done):
        s = to_torch(s)
        a = to_torch(a)
        r = to_torch(r)
        s_next = to_torch(s_next)
        done = to_torch(done)

        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = s_next
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, max_backtrack_len=None):
        idx = np.random.randint(0, self.size, size=batch_size)
        if max_backtrack_len is None:
            # 采样单步数据
            return {
                's': self.s[idx],
                'a': self.a[idx],
                'r': self.r[idx],
                's_next': self.s_next[idx],
                'done': self.done[idx]
            }
        else:
            # 采样多步数据：随机选择回溯长度k（1~max_backtrack_len）
            k = np.random.randint(1, max_backtrack_len + 1)
            action_seqs = []
            s_tk = []
            r_tk = []
            for i in range(batch_size):
                action_seq = []
                for j in range(k):
                    action_seq.append(self.a[(idx[i] + j) % self.size].unsqueeze(0))
                action_seqs.append(torch.cat(action_seq, dim=0))  # (k, action_dim)
                s_tk.append(self.s[(idx[i] + k) % self.size])  # (state_dim)
                r_tk.append(self.r[(idx[i] + k) % self.size])  # (1, )

            return {
                's_t': torch.stack(s_tk),  # (batch, state_dim)
                'action_seq': torch.stack(action_seqs),  # (batch, k, action_dim)
                's_tk': torch.stack(s_tk),  # (batch, state_dim)
                'r_tk': torch.stack(r_tk),  # (batch, 1)
            }
        
    def clear(self):
        self.ptr = 0
        self.size = 0
    
### Utilities ###
def to_torch(data):
    if torch.is_tensor(data):
        return data.float()
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float)
    else:
        raise ValueError(f"Unsupported data type {data} for conversion to torch tensor.")

def load_data(file_path, buffer):
    data = pd.read_csv(file_path)
    unique_indices = data['index'].unique()
    for idx in tqdm(unique_indices, desc="Loading data into buffer", ncols=100):
        traj_data = data[data['index'] == idx]
        # 提取obs和action
        obs_cols = ['obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5']  # obs_0 到 obs_5
        action_cols = ['action_1', 'action_2', 'action_3']

        for i in range(len(traj_data) - 1):
            state = np.array(traj_data.iloc[i][obs_cols].values)
            next_state = np.array(traj_data.iloc[i + 1][obs_cols].values)
            action = np.array(traj_data.iloc[i][action_cols].values)
            reward = np.array(traj_data.iloc[i]['reward'])
            done = np.zeros(1)

            buffer.add(state, action, next_state, reward, done)

        # 处理最后一个时间步
        state = np.array(traj_data.iloc[-1][obs_cols].values)
        next_state = state  # 最后一个时间步的下一个状态与当前状态相同
        action = np.array(traj_data.iloc[-1][action_cols].values)
        reward = np.array(traj_data.iloc[-1]['reward'])
        done = np.ones(1)
        buffer.add(state, action, next_state, reward, done)

def normalize_data(buffer):
    states = buffer.s[:buffer.size]
    actions = buffer.a[:buffer.size]

    state_mean = states.mean(dim=0, keepdim=True)
    state_std = states.std(dim=0, keepdim=True) + 1e-6
    action_mean = actions.mean(dim=0, keepdim=True)
    action_std = actions.std(dim=0, keepdim=True) + 1e-6

    buffer.s = (buffer.s - state_mean) / state_std
    buffer.s_next = (buffer.s_next - state_mean) / state_std
    buffer.a = (buffer.a - action_mean) / action_std

    logger.info("Data normalization complete.")

def gen_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    real_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    model_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    load_data("./data.csv", real_buffer)
    normalize_data(real_buffer)
    logger.info(f"Real buffer size: {real_buffer.size}")

    # 加载预训练策略
    trainer = ADMPO_OFF(
        state_dim=5,
        action_dim=3,
        policy=policy,
        policy_opt=policy_opt,
        max_backtrack_len=10,
        hidden_dim=516,
        lr_adm=3e-4,
        beta=2.5,
        T_max=5000,
        device=device
    )

    # 训练ADM
    trainer.train_adm(real_buffer, batch_size=512, epochs=10000, std_penalty_coef=0.01)
    trainer.save_adm("./checkpoints/model")


def train_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    real_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    model_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    load_data("./data.csv", real_buffer)
    normalize_data(real_buffer)
    logger.info(f"Real buffer size: {real_buffer.size}")

    # 加载Policy
    policy = Mlp(input_dim=5, hidden_dim=256, output_dim=3)
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)

    # 加载预训练策略
    trainer = ADMPO_OFF(
        state_dim=5,
        action_dim=3,
        policy=policy,
        policy_opt=policy_opt,
        max_backtrack_len=10,
        hidden_dim=516,
        lr_adm=3e-4,
        beta=2.5,
        T_max=5000,
        device=device
    )

    # 加载预训练ADM
    trainer.load_adm("./checkpoints/model")
    # 进行模型rollout并训练策略
    for epoch in tqdm(range(2000), desc="Overall Training"):
        # 重置buffer
        model_buffer.clear()

        # 模型rollout
        initial_states = real_buffer.s[torch.randint(0, real_buffer.size, (256,))]
        trainer.model_rollout(initial_states, model_buffer, rollout_len=10)

        # 训练策略
        loss = trainer.train_policy(real_buffer, model_buffer, batch_size=256, epochs=5)
        trainer.policy_scheduler.step()

        if epoch % 100 == 0:
            print()
            logger.info(f"Epoch {epoch}, Loss: {loss}")

    trainer.save_policy("./checkpoints/model")
    logger.info("Training complete, models saved.")


def eval_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    real_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    model_buffer = ReplayBuffer(state_dim=5, action_dim=3, max_size=int(1e6))
    load_data("./data.csv", real_buffer)
    normalize_data(real_buffer)
    logger.info(f"Real buffer size: {real_buffer.size}")

    # 加载Policy
    policy = Mlp(input_dim=5, hidden_dim=256, output_dim=3)
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)

    # 加载预训练策略
    trainer = ADMPO_OFF(
        state_dim=5,
        action_dim=3,
        policy=policy,
        policy_opt=policy_opt,
        max_backtrack_len=10,
        hidden_dim=516,
        lr_adm=3e-4,
        beta=2.5,
        T_max=5000,
        device=device
    )

    # 加载预训练ADM和Policy
    trainer.load_policy("./checkpoints/model")
    eval_loss = trainer.evaluate_policy(real_buffer, batch_size=256)
    logger.info(f"Evaluation Loss: {eval_loss}")

if __name__ == "__main__":
    # train_main()
    eval_main()
    # gen_main()


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader


# -----------------------
# 定义三层 MLP
# -----------------------
class Mlp(nn.Module):
    def __init__(self,
                 input_dim: int = 150,
                 hidden_dim1: int = 256,
                 hidden_dim2: int = 256,
                 output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------
# 训练和验证函数
# -----------------------
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    for obs, act in loader:
        obs = obs.to(device)
        act = act.to(device)

        pred = model(obs)
        loss = F.mse_loss(pred, act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * obs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for obs, act in loader:
        obs = obs.to(device)
        act = act.to(device)
        pred = model(obs)
        loss = F.mse_loss(pred, act)
        total_loss += loss.item() * obs.size(0)
    return total_loss / len(loader.dataset)


# -----------------------
# 主程序
# -----------------------
def main():
    data = pd.read_csv("./data.csv")

    unique_indices = data['index'].unique()
    obs_list = []
    action_list = []

    for idx in unique_indices:
        traj_data = data[data['index'] == idx]

        # 提取obs和action
        obs_cols = ['obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5']  # obs_0 到 obs_5
        action_cols = ['action_1', 'action_2', 'action_3']

        obs = traj_data[obs_cols].values
        action = traj_data[action_cols].values

        # 处理历史obs拼接
        T = len(obs)

        for t in range(T - 1):
            if t < 29:
                # 对于前30个时间步，复制第一帧的obs
                # 这里的做法是使用第一帧进行填充
                history = np.tile(obs[0], (30 - t - 1, 1))
                current = obs[:t + 1]
                history_obs = np.concatenate([history, current]).flatten()
            else:
                # 对于后面的时间步，使用前30个时间步的obs
                history_obs = obs[t - 29:t + 1].flatten()

            obs_list.append(history_obs)
            action_list.append(action[t + 1])

    obs_tensor = torch.from_numpy(np.array(obs_list)).float()
    act_tensor = torch.from_numpy(np.array(action_list)).float()
    # 构建 Dataset
    dataset = TensorDataset(obs_tensor, act_tensor)

    # 划分训练/验证集：90% 训练，10% 验证
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoader
    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mlp().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 用来追踪效果最好的验证损失
    best_val_loss = float('inf')
    best_model_path = "checkpoints/model_baseline.pth"

    # 训练与验证循环
    epochs = 2000
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 如果当前验证集损失更好，则保存到 CPU 并序列化
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 将模型移动到 CPU
            model_cpu = model.to("cpu")
            # 保存 state_dict
            torch.save(model_cpu.state_dict(), best_model_path)
            print(f"  -> New best model saved (val_loss: {best_val_loss:.6f}) to `{best_model_path}`")
            # 再次将模型移动回训练设备
            model = model_cpu.to(device)

    print(f"训练结束，最优验证损失: {best_val_loss:.6f}")
    print(f"最佳模型已保存为 `{best_model_path}` （保存在 CPU 模式下的 state_dict）")

    # 如果需要加载最佳模型到 CPU 并做推理，可以使用：
    # best_model = Mlp()
    # best_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    # best_model.eval()


if __name__ == "__main__":
    main()
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# CriticNetwork: 用于估计动作值 Q(s, a)
# 说明：
# - 输入：全局状态（state）和所有 agent 的动作（action），拼接后作为网络输入
# - 输出：一个标量 Q 值
# - 该网络用于 MADDPG 中的 critic 训练
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                 n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 全局状态维度 + 所有 agent 的动作维度
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # 优化器和学习率调度
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)

        # 自动检测设备
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # 前向传播：先把 state 与 action 在特征维度拼接，再过两层隐藏层，最后输出 Q 值
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        # 保存模型参数到文件（目录不存在则创建）
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # 从磁盘加载参数（映射到 CPU，避免 GPU 错误）
        print(f'...loading critic checkpoint from {self.chkpt_file}...')
        self.load_state_dict(
            T.load(self.chkpt_file, map_location=T.device('cpu'))
        )


# ActorNetwork: 策略网络，输出每个 agent 的动作
# 输入：该 agent 的局部观测 state
# 输出：动作向量，使用 Softsign 激活将输出限制在 [-1, 1]
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)

        # 自动检测设备
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # 前向传播：两层全连接 + leaky_relu，然后输出经过 Softsign 映射到 [-1,1]
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x))  # 输出动作范围 [-1,1]
        return pi

    def save_checkpoint(self):
        # 保存 actor 参数
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # 从磁盘加载参数（映射到 CPU，避免 GPU 错误）
        print(f'...loading actor checkpoint from {self.chkpt_file}...')
        self.load_state_dict(
            T.load(self.chkpt_file, map_location=T.device('cpu'))
        )

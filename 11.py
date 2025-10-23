import argparse, math, random, os
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

# ===========================
# 一、全局设置与随机种子
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {DEVICE}")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
#经验回放的 Transition 元组
Transition = namedtuple("Transition", ("obs", "act", "rew", "next_obs", "done"))


# ===========================
# 二、可视化工具类
# ===========================
# 成功率的记录与绘制
class Visualizer:
    def __init__(self):
        self.train_rewards = []  # 记录每个episode的奖励
        self.success_rates = []  # 用于记录成功率
        plt.style.use('default') # 设置默认样式
        plt.rcParams.update({
            'figure.facecolor': 'white', 'axes.grid': True,
            'grid.alpha': 0.3, 'lines.linewidth': 2
        })
        self.fig, self.ax = None, None

    def update_metrics(self, reward, success):# 更新奖励和成功率
        self.train_rewards.append(reward)
        self.success_rates.append(1 if success else 0)

    def plot_training_metrics(self, window=100, save_path="training_metrics.png"):
        plt.figure(figsize=(12, 4))
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 平滑处理函数
        def moving_average(data, w):
            if len(data) < w:
                return np.array(data)
            return np.convolve(data, np.ones(w)/w, mode='valid')

        # 奖励曲线
        rewards = moving_average(self.train_rewards, window)
        ax1.plot(rewards, 'b-', label='Reward', alpha=0.8)
        ax1.set_title(f"Average Reward (window={window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)
        ax1.legend()

        # 成功率曲线
        success_rates = moving_average(self.success_rates, window)
        ax2.plot(success_rates, 'g-', label='Success Rate', alpha=0.8)
        ax2.set_title(f"Success Rate (window={window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Rate")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')

    def init_live_display(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-1, 1);
        self.ax.set_ylim(-1, 1)
        plt.show()

    def update_live_display(self, env, ep_reward=0):
        if self.fig is None: self.init_live_display()
        self.ax.clear()
        self.ax.set_xlim(-1, 1);
        self.ax.set_ylim(-1, 1)

        # 防御区
        self.ax.scatter(env.defense_zone[0], env.defense_zone[1], c='g', marker='s', s=200, label='Defense Zone',
                        zorder=1)
        # 障碍物
        for (cx, cy), r in env.obstacles:
            self.ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.4, zorder=0))

        # 目标
        self.ax.scatter(env.target_pos[0], env.target_pos[1], c='r', marker='*', s=200, label='Target', zorder=3)
        # 智能体
        self.ax.scatter(env.agents_pos[:, 0], env.agents_pos[:, 1], c='b', marker='^', s=100, label='Agents', zorder=3)
        # 速度向量
        self.ax.quiver(env.agents_pos[:, 0], env.agents_pos[:, 1], env.agents_vel[:, 0], env.agents_vel[:, 1],
                       color='cyan', scale=5.0, headwidth=4)

        # 可见性连接
        for i, j in self._get_connections_from_state(env):
            self.ax.plot([env.agents_pos[i, 0], env.agents_pos[j, 0]], [env.agents_pos[i, 1], env.agents_pos[j, 1]],
                         'g-', alpha=0.3, zorder=2)

        self.ax.grid(True)
        self.ax.legend(loc='upper right')
        self.ax.set_title(f"Step: {env.t} | Reward: {ep_reward:.2f}")
        plt.pause(0.01)

    def _get_connections_from_state(self, env):# 获取当前智能体间的可见连接
        connections = []
        for i in range(env.N):
            for j in range(i + 1, env.N):
                if not env._is_blocked(env.agents_pos[i], env.agents_pos[j]):
                    connections.append((i, j))
        return connections


# ===========================
# 三、环境定义：InterceptEnv
# ===========================
class InterceptEnv:
    def __init__(self, N=3, M=20, area_size=1.0, max_steps=100, kappa=1.3):
        self.N, self.M, self.area = N, M, area_size
        self.max_steps = max_steps  # 论文中每个episode是100步
        self.step_dt = 0.1  # 论文中的仿真时间步长
        self.kappa = kappa
        self.target_max_speed = 0.25  # 论文中目标的最大速度
        self.agent_max_speed = self.target_max_speed * kappa  # 追踪智能体速度是目标的κ倍
        self.reset()

    def reset(self):
        # 按照论文中的初始化范围设置
        self.defense_zone = np.array([np.random.uniform(0.7, 0.9), np.random.uniform(0.7, 0.9)])
        self.target_pos = np.array([np.random.uniform(-0.95, -0.75), np.random.uniform(-0.95, -0.75)])
        self.target_vel = np.zeros(2)  # 初始速度为0
        self.agents_pos = np.random.uniform(-0.5, 0.5, (self.N, 2))
        self.agents_vel = np.zeros((self.N, 2))  # 初始速度为0
        self.obstacles = [(np.random.uniform(-0.9, 0.9, 2), np.random.uniform(0.05, 0.12)) for _ in range(self.M)]
        self.t = 0
        return self._get_all_obs()


    def _is_blocked(self, pos1, pos2):
        # [修正] 使用更鲁棒的线段-圆相交检测
        p1, p2 = np.array(pos1), np.array(pos2)
        for obs_pos, r in self.obstacles:
            obs = np.array(obs_pos)
            if np.linalg.norm(p2 - p1) < 1e-9: return False  # 两个点重合

            vec_line = p2 - p1
            vec_obs = obs - p1

            t = np.dot(vec_obs, vec_line) / np.dot(vec_line, vec_line)
            t = np.clip(t, 0, 1)  # 将投影点限制在线段上

            closest_point = p1 + t * vec_line
            if np.linalg.norm(closest_point - obs) <= r:
                return True
        return False

    def _get_obs_for(self, i):# 获取第 i 个智能体的观测
        obs = np.zeros(4 + 4 + 4 * (self.N - 1))
        obs[0:2], obs[2:4] = self.agents_pos[i], self.agents_vel[i]

        if not self._is_blocked(self.agents_pos[i], self.target_pos):
            obs[4:6] = self.target_pos - self.agents_pos[i]
            obs[6:8] = self.target_vel - self.agents_vel[i]  # 使用相对速度

        idx = 8
        for j in range(self.N):
            if i == j: continue
            if not self._is_blocked(self.agents_pos[i], self.agents_pos[j]):
                obs[idx:idx + 2] = self.agents_pos[j] - self.agents_pos[i]
                obs[idx + 2:idx + 4] = self.agents_vel[j] - self.agents_vel[i]  # 使用相对速度
            idx += 4
        return obs

    def _get_all_obs(self):
        return np.stack([self._get_obs_for(i) for i in range(self.N)])

    def step(self, actions):
        self.t += 1
        self.agents_vel = np.clip(actions, -1, 1) * self.agent_max_speed
        self.agents_pos += self.agents_vel * self.step_dt

        dir_to_defense = self.defense_zone - self.target_pos
        dist = np.linalg.norm(dir_to_defense)
        if dist > 1e-6: self.target_vel = dir_to_defense / dist * self.target_max_speed
        self.target_pos += self.target_vel * self.step_dt

        dists_to_target = np.linalg.norm(self.agents_pos - self.target_pos, axis=1)  # 智能体到目标的距离

        success = np.any(dists_to_target < 0.1)  # 判断是否成功拦截目标
        target_reached_defense = np.linalg.norm(self.target_pos - self.defense_zone) < 0.1  # 目标是否到达防御区
        timeout = self.t >= self.max_steps  # 是否达到最大步数

        done = success or target_reached_defense or timeout  # 判断是否结束

        return self._get_all_obs(), dists_to_target, success, done



class DAActor(nn.Module):
    """方向辅助Actor（Direction-Assisted）"""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
        self.extr_fc = nn.Linear(4, 64)  # 单独提取目标信息部分（obs[4:8]）
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        h = self.relu(self.fc2(x))
        # 单独提取目标相对位置/速度
        extr = self.relu(self.extr_fc(obs[:, 4:8]))
        h = h + extr  # 融合方向辅助信息
        return self.tanh(self.fc3(h))  # 输出动作范围 [-1,1]


class DPFCritic(nn.Module):
    """维度金字塔融合Critic（完整版）"""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.relu(self.norm2(self.fc2(x)))
        x, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.relu(self.norm3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args): self.buffer.append(Transition(*args))

    def sample(self, batch_size): return Transition(*zip(*random.sample(self.buffer, batch_size)))

    def __len__(self): return len(self.buffer)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 优先学习重要经验"""
    def __init__(self, capacity=int(1e6), alpha=0.6, beta=0.4):
        self.buffer = []
        self.capacity = capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样权重
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # 新经验给予最大优先级
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # 根据优先级计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return Transition(*zip(*samples)), indices, weights

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class MALFAC:# 多智能体方向辅助演员-维度金字塔融合评论家算法
    def __init__(self, N, obs_dim, act_dim, lr=1e-3, gamma=0.99, tau=0.01,
                 alpha=1e-3, beta=1e-3, batch_size=1024):  # 按论文修改batch size为1024
        self.N, self.obs_dim, self.act_dim = N, obs_dim, act_dim
        self.gamma, self.tau = gamma, tau
        self.alpha = alpha  # 论文中设置为1e-3
        self.beta = beta   # 论文中设置为1e-3
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(int(1e6))  # 论文中设置为10^6

        self.actors = [DAActor(obs_dim, act_dim).to(DEVICE) for _ in range(N)]
        self.actors_target = [DAActor(obs_dim, act_dim).to(DEVICE) for _ in range(N)]
        # Critic输入维度是所有智能体 obs 和 act 拼接
        critic_input_dim = (obs_dim + act_dim) * N
        self.critics = [DPFCritic(critic_input_dim).to(DEVICE) for _ in range(N)]
        self.critics_target = [DPFCritic(critic_input_dim).to(DEVICE) for _ in range(N)]

        for i in range(N):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.optA = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.optC = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

    def select_actions(self, obs_all, noise_std=0.1):
        acts = []
        for i in range(self.N):
            o = torch.FloatTensor(obs_all[i:i + 1]).to(DEVICE)
            with torch.no_grad():
                a = self.actors[i](o).cpu().numpy().squeeze(0)
            a += noise_std * np.random.randn(self.act_dim)
            acts.append(np.clip(a, -1, 1))
        return np.stack(acts)

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.push(obs, act, rew, next_obs, float(done))

    def soft_update(self, net, net_tgt):
        for p, p_t in zip(net.parameters(), net_tgt.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

    def train_step(self):#
        if len(self.buffer) < self.batch_size: return
        B = self.batch_size
        trans = self.buffer.sample(B)
        obs_b = torch.FloatTensor(np.stack(trans.obs)).to(DEVICE)
        act_b = torch.FloatTensor(np.stack(trans.act)).to(DEVICE)
        rew_b = torch.FloatTensor(np.stack(trans.rew)).to(DEVICE)
        next_obs_b = torch.FloatTensor(np.stack(trans.next_obs)).to(DEVICE)
        done_b = torch.FloatTensor(trans.done).unsqueeze(1).to(DEVICE)

        obs_flat = obs_b.view(B, -1)
        act_flat = act_b.view(B, -1)
        next_obs_flat = next_obs_b.view(B, -1)

        for i in range(self.N):
            with torch.no_grad():
                next_acts = [self.actors_target[j](next_obs_b[:, j, :]) for j in range(self.N)]
                next_act_flat = torch.cat(next_acts, dim=1)
                y = rew_b[:, i] + (1 - done_b.squeeze()) * self.gamma * \
                    self.critics_target[i](torch.cat([next_obs_flat, next_act_flat], dim=1))

            q = self.critics[i](torch.cat([obs_flat, act_flat], dim=1))
            lossC = nn.MSELoss()(q, y.detach())
            self.optC[i].zero_grad();
            lossC.backward();
            self.optC[i].step()

            curr_acts = [self.actors[j](obs_b[:, j, :]) if j == i else self.actors[j](obs_b[:, j, :]).detach() for j in
                         range(self.N)]
            act_flat_now = torch.cat(curr_acts, dim=1)
            q_now = self.critics[i](torch.cat([obs_flat, act_flat_now], dim=1))
            L1 = -q_now.mean()
            L2 = nn.MSELoss()(self.actors[i](obs_b[:, i, :]), act_b[:, i, :])
            L3 = self.actors[i](obs_b[:, i, :]).norm(p=2, dim=1).mean()
            lossA = L1 + self.alpha * L2 + self.beta * L3

            self.optA[i].zero_grad();
            lossA.backward();
            self.optA[i].step()

            self.soft_update(self.actors[i], self.actors_target[i])
            self.soft_update(self.critics[i], self.critics_target[i])


# ===========================
# 七、训练与测试函数
# ===========================
def train(args):
    env = InterceptEnv(N=args.N, M=args.M, kappa=args.kappa)
    obs_dim = env.reset().shape[1]
    agent = MALFAC(args.N, obs_dim, 2, lr=args.lr, batch_size=args.batch)
    vis = Visualizer()

    # 存储训练历史数据用于可视化
    rewards_history = []
    success_rates_history = []
    window_size = args.log

    # 预填充经验回放池
    print("[INFO] 预填充经验回放池...")
    for _ in range(max(1024, args.batch)):
        obs = env.reset()
        for _ in range(50):  # 每个episode最多50步用于预填充
            acts = np.random.uniform(-1, 1, (args.N, 2))  # 随机动作
            next_obs, dists, success, done = env.step(acts)
            individual_reward = -dists
            shared_reward = -np.mean(dists)
            rew = args.lam * individual_reward + (1 - args.lam) * shared_reward
            agent.store(obs, acts, rew, next_obs, done)
            obs = next_obs
            if done: break
    print(f"[INFO] 经验池已填充 {len(agent.buffer)} 条数据")

    pbar = trange(args.episodes, desc="训练中")
    for ep in pbar:
        obs = env.reset()
        ep_reward, success = 0, False

        # 探索噪声衰减：从args.noise线性衰减到0.01
        noise_std = max(0.01, args.noise * (1 - ep / args.episodes))

        while True:
            acts = agent.select_actions(obs, noise_std=noise_std)
            next_obs, dists, success, done = env.step(acts)

            # [奖励函数逻辑] - 恢复到简单但有效的版本
            individual_reward = -dists  # 个体奖励：到目标的距离的负值
            shared_reward = -np.mean(dists)  # 共享奖励：所有智能体到目标的平均距离的负值

            # 增加成功拦截的额外奖励
            if success:
                success_bonus = 10.0  # 成功拦截给予大额奖励
                individual_reward = individual_reward + success_bonus
                shared_reward = shared_reward + success_bonus

            # 如果目标到达防御区，给予惩罚
            target_reached_defense = np.linalg.norm(env.target_pos - env.defense_zone) < 0.1
            if target_reached_defense:
                penalty = -10.0
                individual_reward = individual_reward + penalty
                shared_reward = shared_reward + penalty

            rew = args.lam * individual_reward + (1 - args.lam) * shared_reward  # 总奖励

            agent.store(obs, acts, rew, next_obs, done)
            obs = next_obs
            ep_reward += np.mean(rew)  # 累加奖励

            agent.train_step()
            if done: break

        vis.update_metrics(ep_reward, success)
        rewards_history.append(ep_reward)
        success_rates_history.append(1.0 if success else 0.0)

        if (ep + 1) % args.log == 0:
            avg_rew = np.mean(rewards_history[-window_size:])
            avg_succ = np.mean(success_rates_history[-window_size:])
            pbar.set_description(f"Ep {ep + 1} | Avg Reward: {avg_rew:.2f} | Avg Success: {avg_succ:.2f} | Noise: {noise_std:.3f}")
            # 保存训练曲线
            vis.plot_training_metrics(window=window_size)

        if (ep + 1) % args.save == 0:
            torch.save(agent, f"malfac_ckpt_{ep + 1}.pth")

    # 训练结束后生成并保存最终的可视化结果
    plot_final_metrics(rewards_history, success_rates_history, window=window_size)

    # 输出最终统计
    final_success_rate = np.mean(success_rates_history[-50:]) if len(success_rates_history) >= 50 else np.mean(success_rates_history)
    print(f"\n[INFO] 训练完成！")
    print(f"[INFO] 最后50轮平均成功率: {final_success_rate:.2%}")
    print(f"[INFO] 总体平均成功率: {np.mean(success_rates_history):.2%}")
    print(f"[INFO] 可视化结果已保存")


def test_ep(env, agent, vis, ep_reward=0.0):
    """用于 live display 的单回合测试"""
    obs = env.reset()
    vis.init_live_display()
    while True:
        acts = agent.select_actions(obs, noise_std=0)  # 无噪声
        obs, _, _, done = env.step(acts)
        vis.update_live_display(env, ep_reward)
        if done: break


def plot_final_metrics(rewards, success_rates, window=50, save_path="training_metrics.png"):
    """绘制并保存最终的训练指标可视化图表

    Args:
        rewards: 训练过程中每个episode的奖励列表
        success_rates: 训练过程中每个episode的成功率列表
        window: 滑动平均窗口大小
        save_path: 保存图表的路径
    """
    # 设置样式
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.size': 12
    })

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 平滑处理函数
    def moving_average(data, w):
        if len(data) < w:
            return np.array(data), np.arange(len(data))
        smoothed = np.convolve(data, np.ones(w)/w, mode='valid')
        return smoothed, np.arange(w-1, len(data))

    # 奖励曲线 - 原始数据+平滑曲线
    raw_x = np.arange(len(rewards))
    ax1.plot(raw_x, rewards, 'b-', alpha=0.3, label='Raw Rewards')
    smoothed_rewards, smooth_x = moving_average(rewards, window)
    ax1.plot(smooth_x, smoothed_rewards, 'b-', linewidth=2, label=f'Moving Avg (window={window})')

    # 添加标题和标签
    ax1.set_title("Training Rewards", fontsize=14)
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 成功率曲线 - 原始数据+平滑曲线
    ax2.plot(raw_x, success_rates, 'g-', alpha=0.3, label='Raw Success Rate')
    smoothed_success, smooth_x = moving_average(success_rates, window)
    ax2.plot(smooth_x, smoothed_success, 'g-', linewidth=2, label=f'Moving Avg (window={window})')

    # 添加标题和标签
    ax2.set_title("Interception Success Rate", fontsize=14)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Success Rate", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图片（不关闭窗口）
    plt.show(block=True)  # 阻塞模式，确保图片不会立即关闭
    print(f"[INFO] 训练指标可视化已保存至 {save_path}")


# ===========================
# 八、主函数入口
# ===========================
if __name__ == "__main__":
    # 确保必要的导入
    import argparse
    import torch
    import matplotlib.pyplot as plt

    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=500)    # 训练回合数，增加到500轮以获得更好的训练效果
    parser.add_argument("--max_steps", type=int, default=100)    # 每个episode的最大步数
    parser.add_argument("--log", type=int, default=10)           # 日志间隔
    parser.add_argument("--save", type=int, default=100)         # 保存间隔

    # 环境参数 (按论文设置)
    parser.add_argument("--N", type=int, default=3)              # 智能体数量 {3,4,5}
    parser.add_argument("--M", type=int, default=20)             # 障碍物数量 {15,20,25}
    parser.add_argument("--kappa", type=float, default=1.3)      # 速度比例系数 {1,1.3,1.6}

    # 训练参数 (按论文设置)
    parser.add_argument("--lr", type=float, default=1e-3)        # Adam优化器学习率
    parser.add_argument("--batch", type=int, default=1024)       # 批次大小
    parser.add_argument("--noise", type=float, default=0.1)      # 探索噪声
    parser.add_argument("--lam", type=float, default=0.9)        # 奖励混合系数
    parser.add_argument("--live_display", action="store_true", help="是否在训练时进行实时可视化")

    # 解析命令行参数
    args = parser.parse_args()

    try:
        if args.mode == "train":
            print("[INFO] 开始训练...")
            train(args)
            print("[INFO] 训练完成!")
        else:
            print("[INFO] 开始测试...")
            env = InterceptEnv(N=args.N, M=args.M, kappa=args.kappa)
            agent = torch.load("malfac_ckpt_1000.pth", map_location=DEVICE)
            vis = Visualizer()
            test_ep(env, agent, vis)
            print("[INFO] 测试完成!")
    except Exception as e:
        print(f"[ERROR] {str(e)}")

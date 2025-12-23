import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np


# Agent：MADDPG 中每个智能体对应的类，封装了 actor/critic 网络、目标网络、动作选择和参数更新逻辑
# 该类负责：
# - 构建 actor/critic 及其 target 网络
# - 在训练/评估时根据 actor 输出选择动作（带可选噪声用于探索）
# - 对 target 网络做软更新（或者在初始化时做硬更新）
# - 保存/加载模型参数
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        # 参数说明（中文）：
        # actor_dims: 当前 agent 的观测向量维度（int）
        # critic_dims: critic 的输入维度（通常是所有 agent 的观测维度之和）
        # n_actions: 每个 agent 的动作维度（例如 2 表示二维连续动作）
        # n_agents: 多智能体数量（用于 critic 的网络结构）
        # agent_idx: 当前 agent 的索引（用于命名和检查点管理）
        # chkpt_dir: 模型保存/加载的文件夹根路径
        # alpha: actor 学习率（或 actor 网络的相关超参数）
        # beta: critic 学习率
        # fc1, fc2: 网络中隐藏层的宽度
        # gamma: 折扣因子（环境回报折扣）
        # tau: target 网络软更新的系数（tau=1 表示直接拷贝）

        # 保存常用超参数
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        # 给 agent 一个可读的名称（用于命名 checkpoint 文件等）
        self.agent_name = 'agent_%s' % agent_idx

        # 构建 actor / critic / target_actor / target_critic 网络
        # ActorNetwork / CriticNetwork 已在 networks.py 中实现（假定包含 save/load/checkpoint 等方法）
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        # 初始化时让 target 网络与 online 网络参数一致（tau=1，硬拷贝）
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        """
        根据当前 actor 网络输出选择动作，并在训练时加入噪声以便探索。

        输入：
        - observation: 当前 agent 的观测（numpy 向量）
        - time_step: 全局步数（用于噪声衰减）
        - evaluate: 是否为评估模式（True 时不加噪声）

        返回：
        - action_np: numpy 数组形式的动作（形状 (n_actions,)）

        说明与实现细节：
        - 首先把 observation 转为 torch tensor 并送入 actor 网络得到基础动作（通常是未经裁剪的连续输出）
        - 加入均匀噪声 noise ∈ [-1, 1]，并按噪声尺度 noise_scale 缩放。
          noise_scale 随时间衰减（使用指数衰减 decay_rate），并在训练中从 max_noise 逐渐下降到 min_noise。
        - 若 evaluate=True，则不添加噪声（保证策略确定性），即 noise 被置为 0
        - 最后将动作从 tensor 转为 numpy，并对其幅值进行裁剪：若动作向量模长大于 0.04，则按比例缩放到 0.04
          （这是任务层面人为限制，保证动作幅度不会过大）
        """

        # 把 observation 包装为 batch size 1 的 torch tensor，并放到 actor 网络的 device 上
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)

        # exploration 噪声参数设定（超参数）
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        # 随时间衰减的噪声尺度：从 max_noise 以指数形式衰减，保证至少为 min_noise
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))

        # 生成均匀噪声向量，范围 [-1, 1)
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1  # [-1,1)
        if not evaluate:
            # 训练模式：按当前噪声尺度缩放
            noise = noise_scale * noise
        else:
            # 评估模式：不添加噪声
            noise = 0 * noise

        # actor 的输出加上噪声作为最终动作（在 CPU 上转为 numpy 之前是 torch tensor）
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]

        # 对动作做幅度限制：保证动作向量的 L2 范数不超过 0.04（任务级约束）
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    def update_network_parameters(self, tau=None):
        """
        更新 target 网络参数：使用软更新（Polyak averaging）或硬更新（当 tau=1 时直接拷贝）。

        参数：
        - tau: 软更新系数，若为 None 则使用构造函数中 self.tau

        算法：
        target_param = tau * online_param + (1 - tau) * target_param
        这个操作对 actor 和 critic 的 target 网络都执行一次。
        """
        if tau is None:
            tau = self.tau

        # --- 更新 target_actor ---
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        # 把迭代器转换为字典以便逐参数按名字更新
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            # 注意：先 clone 保持张量独立，然后使用软更新公式
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                    (1 - tau) * target_actor_state_dict[name].clone()

        # 把计算得到的新参数加载进 target_actor
        self.target_actor.load_state_dict(actor_state_dict)

        # --- 更新 target_critic ---
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                    (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        # 把 actor/critic 及其 target 的模型参数保存到磁盘（调用网络对象的 save_checkpoint）
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        # 从磁盘加载 actor/critic 及其 target 的模型参数（注意：通常 target 模型也会被加载为其 checkpoint）
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

# MADDPG: 多智能体 DDPG 框架封装
# 说明：该类管理多个 Agent 对象（每个 agent 含 actor/critic/target 等），并负责
# - 保存/加载所有 agent 的模型
# - 在评估/运行时调用各 agent 的 choose_action
# - 在训练时根据回放缓冲区采样并更新所有 agent 的网络参数
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/', verbose=True):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.verbose = verbose
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        # 根据智能体数量构建每个 agent（Agent 会创建 actor/critic/target 网络）
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        # 保存所有 agent 的模型参数到 chkpt_dir
        if self.verbose:
            print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        # 从磁盘加载所有 agent 的参数
        if self.verbose:
            print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        # 在运行/评估阶段被调用：对每个 agent 调用 agent.choose_action 并返回动作列表
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        # 在训练阶段被周期性调用：从回放缓冲区采样批次并对所有 agent 执行一轮 actor/critic 的更新
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        # 把 numpy 批次转换为 torch tensor
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
    
        # 对每个 agent，使用其 target_actor 计算下一个动作（用于 critic 的 target 计算）
        for agent_idx, agent in enumerate(self.agents):

            new_states = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        # 把各 agent 的动作按列拼接以构建 critic 需要的动作向量
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        # 对每个 agent 分别计算 critic_loss（MSE）和 actor_loss
        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            agent.update_network_parameters()

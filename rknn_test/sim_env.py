import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy


# UAVEnv: 一个简单的无人机（UAV）围捕/包围目标的仿真环境
# 说明：该环境实现了多个 UAV（agent）与一个目标（target）的动力学更新、传感器（激光束）测距、
# 障碍物动态、奖励与终止条件，以及用于可视化的 render()/render_anime() 方法。
class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        self.length = length  # 边界长度
        self.num_obstacle = num_obstacle  # 障碍物数量
        self.num_agents = num_agents  # 智能体数量

        # 物理参数与约束
        self.time_step = 0.5  # 仿真时间步长（积分步长）
        self.v_max = 0.1  # 普通 UAV 的最大线速度
        self.v_max_e = 0.12  # 目标（或敌方）的最大线速度
        self.a_max = 0.04  # 最大加速度
        self.a_max_e = 0.05 # 目标最大加速度

        # 传感器与感知相关
        self.L_sensor = 0.2  # 激光传感器的最大探测距离
        self.num_lasers = 16  # 激光束数目（环形扫描）
        # multi_current_lasers: 每个 agent 的激光距离数组（初始化为最大距离）
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]

        # 智能体名字（用于索引或调试）
        self.agents = ['agent_0', 'agent_1', 'agent_2', 'target']

        # 随机种子/环境内部状态
        self.info = np.random.get_state()

        # 障碍物列表：每个 obstacle 的初始位置、速度与半径在 obstacle() 中随机生成
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]

        # 历史位置，用于绘制轨迹；每个智能体一个列表
        self.history_positions = [[] for _ in range(num_agents)]

        # action_space / observation_space：使用 gymnasium 的 spaces 描述尺寸（不是严格约束）
        # action: 每个 agent 的动作为二维连续向量 [a_x, a_y]
        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        }

        # observation_space: 为方便外部使用，给出每个 agent 的观测维度
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }


    def reset(self):
        # 重置环境到一个随机初始状态
        # 返回：multi_obs（list），包含每个智能体的观测向量（flatten 后）
        SEED = random.randint(1, 1000)
        random.seed(SEED)

        # 初始化位置与速度
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            # 非目标智能体（UAV）随机初始化在边界内的一个子区域
            if i != self.num_agents - 1:  # UAV
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:  # target: 目前固定放在较远的位置
                self.multi_current_pos.append(np.array([0.5, 1.75]))
            # 初始速度为 0
            self.multi_current_vel.append(np.zeros(2))

        # 更新激光传感器读数（并检测是否碰撞），以保证初始观测一致
        self.update_lasers_isCollied_wrapper()

        # 组合所有智能体的观测并返回
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):
        # 按照给定的动作更新环境状态
        # 参数 actions: list，长度为 num_agents，每个元素是 [a_x, a_y]
        # 返回：multi_next_obs, rewards, dones
        last_d2target = []  # 记录每个 UAV 到目标的距离（用于奖励计算）

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                # 记录到目标的距离
                last_d2target.append(np.linalg.norm(pos - pos_taget))

            # 根据动作更新速度：简单的显式欧拉积分，速度 += 加速度 * dt
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step

            # 速度裁剪（按智能体类型分别使用不同的最大速度）
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    # 保持速度方向不变，仅缩放到最大速度
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 更新位置：pos += vel * dt
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # 更新障碍物位置（可能会与边界弹回）
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # 边界检查：若碰到边界则反弹并修正位置，保证障碍物留在场内
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        # 更新激光与碰撞信息
        Collided = self.update_lasers_isCollied_wrapper()

        # 根据碰撞信息与上一帧到目标距离计算奖励与 dones
        rewards, dones = self.cal_rewards_dones(Collided, last_d2target)

        # 获取下一时刻的观测
        multi_next_obs = self.get_multi_obs()

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        # 辅助函数：返回每个 agent 的一个简化观测（位置归一化 + 速度归一化）
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max,
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        # 构建每个智能体的观测向量（flatten 后）并返回为列表
        # 观测由：自身位置信息、队友位置信息、激光传感器距离、目标信息 等组成
        total_obs = []
        single_obs = []
        S_evade_d = []  # 仅 target 使用
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]

            # 自身基本信息（位置归一化，速度相对最大速度归一化）
            S_uavi = [pos[0] / self.length, pos[1] / self.length, vel[0] / self.v_max, vel[1] / self.v_max]

            S_team = []  # 记录其他队友的位置
            S_target = []  # 记录目标的相对距离与角度
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    # 加入队友的位置（归一化）
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                elif j == self.num_agents - 1:
                    # 计算到目标的距离与相对角度
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
                    S_target.extend([d / np.linalg.norm(2 * self.length), theta])
                    if i != self.num_agents - 1:
                        # 非 target 的智能体把归一化距离压入逃避距离数组（供 target 观测/策略使用）
                        S_evade_d.append(d / np.linalg.norm(2 * self.length))

            # 激光传感器信息（已经在 update_lasers_isCollied_wrapper 中更新）
            S_obser = self.multi_current_lasers[i]

            if i != self.num_agents - 1:
                single_obs = [S_uavi, S_team, S_obser, S_target]
            else:
                # target 的观测维度略有不同
                single_obs = [S_uavi, S_obser, S_evade_d]

            # 将嵌套列表展开为一维向量并加入 total_obs
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def cal_rewards_dones(self, IsCollied, last_d):
        # 根据当前碰撞信息 IsCollied（每个 agent 的布尔）和上一时刻到目标的距离计算奖励与终止条件
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 奖励项权重（可调整）
        mu1 = 0.7  # 贴近目标的奖励权重
        mu2 = 0.4  # 安全/避撞奖励权重
        mu3 = 0.01  # 多阶段策略奖励权重
        mu4 = 5  # 终结（成功包围）的大额奖励

        d_capture = 0.3  # 捕获距离阈值（小于等于即视为捕获）
        d_limit = 0.75  # 阶段判定阈值

        # 1) 对前三个 UAV 计算接近目标方向的奖励（鼓励向目标靠近且朝向正确）
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)  # 到目标的距离

            # cos_v_d: 用速度方向与指向目标方向的夹角来衡量是否朝向目标移动
            cos_v_d = np.dot(vel, dire_vec) / (v_i * d + 1e-3)
            r_near = abs(2 * v_i / self.v_max) * cos_v_d
            # 把靠近目标的奖励加入（如果方向错或速度小则降低奖励）
            rewards[i] += mu1 * r_near

        # 2) 碰撞/安全相关奖励：如果碰撞则惩罚，否则基于激光最短距离给安全奖励
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu2 * r_safe

        # 3) 多阶段目标（围捕/包围）相关的奖励：基于三角形面积、距离之和等度量
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]

        # 计算若干三角形面积，用来衡量 UAV 与目标的相对布局
        S1 = cal_triangle_S(p0, p1, pe)
        S2 = cal_triangle_S(p1, p2, pe)
        S3 = cal_triangle_S(p2, p0, pe)
        S4 = cal_triangle_S(p0, p1, p2)

        d1 = np.linalg.norm(p0 - pe)
        d2 = np.linalg.norm(p1 - pe)
        d3 = np.linalg.norm(p2 - pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)

        # 3.1 目标 UAV 的奖励：目标希望远离包围者（增大距离），因此奖励为距离增加量的函数
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d), -2, 2)

        # 3.2/3.3/3.4 对队友的多阶段奖励：根据几何关系区分不同阶段（track, encircle, capture）
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = -Sum_d / max([d1, d2, d3])
            rewards[0:2] += mu3 * r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1 / 3 * np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3 * r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1, d2, d3]):
            r_capture = np.exp((Sum_last_d - Sum_d) / (3 * self.v_max))
            rewards[0:2] += mu3 * r_capture

        # 4) 终止/成功判定：若三者完全形成包围并都在捕获距离内，则判为成功并给大额奖励
        if Sum_S == S4 and all(d <= d_capture for d in [d1, d2, d3]):
            rewards[0:2] += mu4 * 10
            dones = [True] * self.num_agents

        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        # 更新每个智能体的激光传感器读数，并判断是否与障碍物碰撞（done 标志）
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                # update_lasers 来自 math_tool.py：计算每个激光束的距离与是否碰撞
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                # 取各障碍物的最小激光距离作为当前激光读数（最近障碍物显著影响观测）
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                # 如果碰撞，将速度置零（停止运动）
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):
        # 绘制当前场景并返回 RGBA 图像（NumPy 数组），便于保存或合成视频
        # 该函数同时把绘图绘制到 Agg canvas，然后把缓冲区转换为数组返回

        plt.clf()

        # 使用图标表示 UAV（png 文件）
        uav_icon = mpimg.imread('UAV.png')

        # 绘制前三个智能体（UAVs）
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])

            # 绘制轨迹线（半透明）
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # 基于速度向量计算飞机图标的旋转角度
            angle = np.arctan2(vel[1], vel[0])

            # 使用仿射变换把图标旋转并移动到 UAV 的当前坐标
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1  # 图标绘制大小（可调整）
            # plt.imshow 的 transform 参数允许图像相对于坐标系旋转和平移
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

            # （可选）激光束可视化代码被注释掉，保留以便调试时使用

        # 绘制目标（Target）及其轨迹
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        # 设置坐标轴范围（加一点 padding）
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()

        # 将当前 Figure 绘制到 Agg canvas 并从缓冲区读取 RGBA 图像
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        # render_anime: 用于创建动画帧（在 FuncAnimation 或主循环中调用）
        plt.clf()

        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)

            trajectory = np.array(self.history_positions[i])
            # 用颜色渐变绘制轨迹，便于观察历史轨迹随时间变化
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1], color=color, alpha=0.7)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        # 绘制目标与轨迹
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        # 关闭图形窗口
        plt.close()


class obstacle():
    def __init__(self, length=2):
        # 障碍物在场地内随机位置生成，速度可为 0（固定障碍）或者随机方向的小速度
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03
        speed = 0.00  # 当前设为固定障碍
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)
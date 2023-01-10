import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 0.001    # actor学习率
LR_CRITIC = 0.002   # critic学习率
GAMMA = 0.9         # 折扣因子
EPISODE = 2000
EP_STEPS = 200      # 每个episode执行200个回合
TAU = 0.01          # 软更新系数
MEMORY_CAPACITY = 10000
RENDER = False
BATCH_SIZE = 32
var = 3             # 噪声的方差
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
# ENV_NAME = 'Ant-v2'
ENV_NAME = 'Humanoid-v3'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]  # 状态维数
a_dim = env.action_space.shape[0]       # 动作维数
a_bound = env.action_space.high     # 动作取值的上下限
a_low_bound = env.action_space.low


class ActorNet(nn.Module):  # 定义actor网络  输入为s，输出为a
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        if ENV_NAME == 'Pendulum-v0':
            actions = x * 2
        elif ENV_NAME == 'Humanoid-v3':
            actions = x * 0.4
        else:
            actions = x
        return actions


class CriticNet(nn.Module):     # 定义critic网络  输入为s和a，输出的是动作的价值
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)

        # critic网络一共有两个输出，对应两个Q函数
        self.out1 = nn.Linear(30, 1)
        self.out1.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(30, 1)
        self.out2.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value1 = self.out1(F.relu(x + y))
        actions_value2 = self.out2(F.relu(x + y))
        return actions_value1, actions_value2   # 输出两个Q值


class DPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        # actor和critic都拥有target网络和policy网络
        self.actor_eval = ActorNet(s_dim, a_dim).to(device)
        self.actor_target = ActorNet(s_dim, a_dim).to(device)
        self.critic_eval = CriticNet(s_dim, a_dim).to(device)
        self.critic_target = CriticNet(s_dim, a_dim).to(device)
        # 均选择Adam优化器
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)

        self.loss_func = nn.MSELoss().to(device)    # 定义损失函数

        self.step = 0   # 记录学习次数，用于决定什么时候更新actor

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # print(transition)
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):  # 选择动作
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
        return self.actor_eval(s)[0].detach()

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    # 从经验回放buffer当中随机选择batch_size个
        batch = self.memory[indices, :]

        batch_state = torch.FloatTensor(batch[:, :self.s_dim]).to(device)
        batch_action = torch.FloatTensor(batch[:, self.s_dim:self.s_dim + self.a_dim]).to(device)
        batch_reward = torch.FloatTensor(batch[:, -self.s_dim - 1: -self.s_dim]).to(device)
        batch_next_state = torch.FloatTensor(batch[:, -self.s_dim:]).to(device)

        self.step += 1

        if self.step % 5 == 0:
            a = self.actor_eval(batch_state)    # 评估action的价值
            q = self.critic_eval(batch_state, a)    # 计算出actor的损失并反向传播
            actor_loss = -torch.mean(q[0])
            # optimize the loss of actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        a_target = self.actor_target(batch_next_state)
        q_tmp1, q_tmp2 = self.critic_target(batch_next_state, a_target)
        q_target = batch_reward + GAMMA * torch.min(q_tmp1, q_tmp2)     # 选择Q值小的来更新

        q_eval1, q_eval2 = self.critic_eval(batch_state, batch_action)
        td_error = self.loss_func(q_target, q_eval1) + self.loss_func(q_target, q_eval2)    # 损失为targetQ值与Q1 Q2的误差

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        # 采用软更新网络权值
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))' + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))' + '.data.add_(TAU*self.critic_eval.' + x + '.data)')


dpg = DPG(a_dim, s_dim, a_bound)

episode = list(range(2000))
episode_bar = tqdm(episode)

for i in episode_bar:
# for i in range(EPISODES):
    state = env.reset()     # 初始化状态
    ep_reward = 0           # 记录一次episode的reward
    for j in range(EP_STEPS):
        if RENDER:          # 渲染环境
            env.render()

        action = dpg.choose_action(state).cpu()     # 选择动作
        action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)   # 加入噪声
        next_state, reward, done, info = env.step(action)   # 执行
        dpg.store_transition(state, action, reward, next_state)     # 将状态存入经验回放buffer当中

        if dpg.pointer > MEMORY_CAPACITY:   # 当经验回放buffer满的时候开始学习
            if var > 0.1:
                var *= 0.9995   # 不断衰减噪声的标准差，但同时还要保证一定的探索度
            dpg.learn()

        state = next_state
        ep_reward += reward
    # print(f'Episode:{i}, Reward:{ep_reward}')
    # f = open('reward_TD_' + ENV_NAME + '.txt', 'a+')
    # f.write(str(ep_r)+'\n')
    # f.close()


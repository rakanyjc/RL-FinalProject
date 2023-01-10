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


class ActorNet(nn.Module):      # 定义Actor网络 输入为s，输出为a
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
            actions = x * 2  # for the game "Pendulum-v0", action range is [-2, 2]
        elif ENV_NAME == 'Humanoid-v3':
            actions = x * 0.4   # for the game "Humanoid-v3", action range is [-0.4, 0.4]
        else:
            actions = x
        return actions


class CriticNet(nn.Module):     # 定义Critic网络 输入为s和a，输出的是动作的价值
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x + y))
        return actions_value


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
        self.step = 0   # 用于更新网络权值

    def store_transition(self, state, action, reward, next_state):  # experience replay buffer 容量为MEMORY_CAPACITY
        transition = np.hstack((state, action [reward], next_state))
        # print(transition)
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):     # 选择动作
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

        a = self.actor_eval(batch_state)
        q = self.critic_eval(batch_state, a)    # 评估action的价值
        actor_loss = -torch.mean(q)             # 计算出actor的损失并反向传播

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        a_target = self.actor_target(batch_next_state)  # 利用target Q与policy Q计算出critic的损失并反向传播
        q_tmp = self.critic_target(batch_next_state, a_target)
        q_target = batch_reward + GAMMA * q_tmp

        q_eval = self.critic_eval(batch_state, batch_action)
        td_error = self.loss_func(q_target, q_eval)
        # print(td_error.detach().cpu().numpy())
        # f = open('loss_DPG_'+ENV_NAME+'.txt', 'a+')
        # f.write(str(td_error.detach().cpu().numpy())+'\n')
        # f.close()

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        self.step += 1
        if self.step % 5 == 0:   # 直接复制权值
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())


dpg = DPG(a_dim, s_dim, a_bound)

episode = list(range(2000))
episode_bar = tqdm(episode)

for i in episode_bar:
# for i in range(EPISODES):
    state = env.reset()     # 初始化状态
    ep_reward = 0           # 记录一次episode的reward
    for j in range(EP_STEPS):
        if RENDER:      # 渲染环境
            env.render()

        action = dpg.choose_action(state).cpu()     # 选择动作
        next_state, reward, done, info = env.step(action)   # 执行
        dpg.store_transition(state, action, reward, next_state)     # 将状态存入经验回放buffer当中

        if dpg.pointer > MEMORY_CAPACITY:   # 当经验回放buffer满的时候开始学习
            dpg.learn()

        state = next_state
        ep_reward += reward
    # print(f'Episode:{i}, Reward:{ep_r}')
    # f = open('reward_DPG_' + ENV_NAME + '.txt', 'a+')
    # f.write(str(ep_r)+'\n')
    # f.close()


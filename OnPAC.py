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

# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
# ENV_NAME = 'Ant-v2'
ENV_NAME = 'Humanoid-v3'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]  # 状态维数
a_dim = env.action_space.shape[0]       # 动作维数
a_bound = env.action_space.high         # 动作取值的上下限
a_low_bound = env.action_space.low


class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)

        self.miu_out = nn.Linear(30, a_dim)
        self.miu_out.weight.data.normal_(0, 0.1)

        self.rou_out = nn.Linear(30, a_dim)
        self.rou_out.weight.data.normal_(0, 0.1)

    def forward(self, x):   # 输出是高斯分布的两个参数
        x = self.fc1(x)
        x = F.relu(x)
        miu = torch.relu(self.miu_out(x)) + 1e-8   # miu > 0   保证miu是大于0的
        rou = self.rou_out(x)                       # sigma^2 = exp(rou)
        # rou = torch.sigmoid(self.rou_out(x))  # sigma^2 = exp(rou)
        # pi = np.exp(-((a-miu)**2) / 2 * (np.exp(rou)))/(np.sqrt(2 * np.pi) * np.sqrt(np.exp(rou)))
        # f = np.log(pi)  # ln_pi
        # return f
        return miu, rou


class CriticNet(nn.Module):     # 定义critic网络
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
        value = self.out(F.relu(x+y))
        return value


class ActorCritic(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.actor = ActorNet(s_dim, a_dim).to(device)
        self.critic = CriticNet(s_dim, a_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.loss_func = nn.MSELoss().to(device)

    def choose_action(self, s):  # normal-distribution
        # s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
        s = torch.tensor(s).to(device).float()
        action = np.clip(np.random.normal(self.actor(s)[0].detach().cpu(), np.sqrt(np.exp(self.actor(s)[1].detach().cpu()))),
                         a_low_bound, a_bound)
        return action

    def learn(self, sample):
        # 采用Sarsa算法来更新权值
        state, action, reward, next_state, next_action = sample
        state = torch.tensor(state).to(device).float()
        action = torch.tensor(action).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        next_state = torch.tensor(next_state).to(device).float()
        next_action = torch.tensor(next_action).to(device).float()

        # train critic network
        td_target = reward + GAMMA * self.critic(next_state, next_action)
        td_error = self.loss_func(td_target.squeeze(), self.critic(state, action).squeeze()).to(device)
        # print(td_error.detach().cpu().numpy())
        # f = open('loss_OnPAC_' + ENV_NAME + '.txt', 'a+')
        # f.write(str(td_error.detach().cpu().numpy()) + '\n')
        # f.close()
        self.critic_optimizer.zero_grad()
        td_error.backward(retain_graph=True)
        self.critic_optimizer.step()

        # train actor network
        miu, rou = self.actor(state)        # 得到高斯分布的两个参数
        # a = np.clip(np.random.normal(miu, np.exp(rou)), a_low_bound, a_bound)   # sample
        ln_pdf = (torch.normal(miu, torch.sqrt(torch.exp(rou))))      # pdf
        actor_loss = torch.mean(ln_pdf * td_error)                    # 损失为概率密度函数乘以td_error的均值
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


Ac = ActorCritic(a_dim, s_dim, a_bound)

episode = list(range(2000))
episode_bar = tqdm(episode)
for i in episode_bar:
# for i in range(EPISODE):
    state = env.reset()     # 初始化状态
    ep_reward = 0           # 记录reward
    # print(f'Episode {i}')
    for j in range(EP_STEPS):
        action = Ac.choose_action(state.transpose())
        next_state, reward, done, _ = env.step(action)
        # print(state, action, reward, next_state.transpose())
        next_action = Ac.choose_action(next_state.transpose())
        sample = state.transpose(), action, reward, next_state.transpose(), next_action
        Ac.learn(sample)
        if ENV_NAME == 'Pendulum-v0':
            ep_reward += reward.squeeze()
        else:
            ep_reward += reward
        state = next_state
    # f = open('reward_OnPAC_' + ENV_NAME + '.txt', 'a+')
    # f.write(str(ep_reward) + '\n')
    # f.close()
    # print(f'Episode:{i}, Reward:{ep_reward}')

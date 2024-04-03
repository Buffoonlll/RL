import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, env, ppo_type, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-2, gamma=0.98, epochs=10, lmbda=0.95,
                 beta=3.0, eps=0.2, kl_constraint=0.0005, num_episodes=500):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.env = env
        self.ppo_type = ppo_type
        state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        action_dim = env.action_space.n
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epochs = epochs
        self.lmbda = lmbda
        self.beta = beta
        self.eps = eps
        self.kl_constraint = kl_constraint
        self.num_episodes = num_episodes
        self.return_list = []
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def take_action(self, state):
        state = torch.tensor(np.array([state])).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def advantage(self, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0
        for delta in td_delta[::-1]:
            advantage = delta + self.gamma * self.lmbda * advantage
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.FloatTensor(advantage_list).to(self.device)

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.advantage(td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states))

        if self.ppo_type == 'ppo-penalty':
            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                new_action_dists = torch.distributions.Categorical(self.actor(states))
                kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists).detach())
                if kl_div > 1.5 * self.kl_constraint:
                    self.beta = self.beta * 2
                elif kl_div < self.kl_constraint / 1.5:
                    self.beta = self.beta / 2
                else:
                    pass
                actor_loss = -torch.mean(ratio * advantage - self.beta * kl_div)
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        elif self.ppo_type == 'ppo-clip':
            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def train(self):
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes / 10)):
                    episode_return = 0
                    transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
                    state = self.env.reset()[0]
                    done = False
                    while not done:
                        action = self.take_action(state)
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        done = done or truncated
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['rewards'].append(reward)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_return += reward
                        self.return_list.append(episode_return)
                        self.update(transition_dict)
                        if (i_episode + 1) % 10 == 0:
                            pbar.set_postfix({
                                'episode': '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                                'return': '%.3f' % np.mean(self.return_list[-10:])
                            })
                        pbar.update(1)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size])[1::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    agents = []
    labels = []
    # for Beta in [0.01, 0.1, 1, 3, 10]:
    #     Agent.append(PPO(Env, 'ppo-penalty', beta=Beta))
    agents.append(PPO(env, 'ppo-penalty'))
    agents.append(PPO(env, 'ppo-clip'))
    episodes_list = []
    returns_list = []
    mv_returns_list = []
    for agent in agents:
        print(agent.ppo_type + ' ' + str(agent.beta))
        if agent.ppo_type == 'ppo-penalty':
            labels.append(agent.ppo_type + ' beta=' + str(agent.beta))
        else:
            labels.append(agent.ppo_type)
        agent.train()
        episodes_list.append(list(range(len(agent.return_list))))
        returns_list.append(agent.return_list)
        mv_returns_list.append(moving_average(agent.return_list, 9))

    plt.figure(1)
    for t in range(len(episodes_list)):
        plt.plot(episodes_list[t], returns_list[t], label=labels[t])
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.legend()
    plt.show()

    plt.figure(2)
    for t in range(len(episodes_list)):
        plt.plot(episodes_list[t], mv_returns_list[t], label=labels[t])
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.legend()
    plt.show()

# Asynchronous Advantage Actor - Critic (A3C)

### Policy Gradient

$$
\nabla{\overline{R_\theta}}\approx\frac{1}{N}\sum_{n=1}^{N}{\sum_{t=1}^{T_n}{(\Sigma^{T_n}_{t'=t}\gamma^{t'-t}r_{t'}^n-b){\nabla}logp_\theta(a_t^n|s_t^n)}}\\
b=baseline
$$

其中：

> $$
> E[G_t^n]=\Sigma^{T_n}_{t'=t}\gamma^{t'-t}r_{t'}^n=Q^{\pi_\theta}(s_t^n,a_t^n)\\
> baseline=V^{\pi_\theta}(s_t^n)
> $$



### Advantage Function

> $A(s,a)=Q(s,a)-V(s)$



### Actor

由状态输出动作$s_1→a_1$

代码：

```python
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
```



### Critic

输入一个动作$\pi$，衡量 $\pi$ 的好坏 $V^\pi(s)$​

代码：

```python
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```



### Asynchronous（异步强化学习）

运用多个agent进行训练~~（鸣人多重影分身）~~

# TRPO（trust region policy optimization）

> 假设$\theta$表示策略$\pi_\theta$的参数，定义
>
> $$J(\theta)=E_{s_0}[V^{\pi_\theta(s_0)}]=E_{\pi_\theta}[\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t)]$$​
>
> 基于策略的方法的目标是找到
> $$
> \theta^*=arg \ max_{\theta}J(\theta)
> $$
> 策略梯度算法主要沿着$\nabla_{\theta}J(\theta)$方向迭代更新策略参数$\theta$。但是这种算法有一个明显的缺点：当策略网络是深度模型时，沿着策略梯度更新参数，很有可能由于步长太长，策略突然显著变差，进而影响训练效果。
>
> 针对以上问题，我们考虑在更新时找到一块**信任区域**（trust region），在这个区域上更新策略时能够得到某种策略性能的安全性保证，这就是TRPO

## 算法流程

- 初始化策略网络参数$\theta$，价值网络参数$\omega$​
- **for**  序列 $e=1{\rightarrow}E$  **do**:
  - 用当前策略$\pi_{\theta}$采样轨迹$\{s_1,a_1,r_1,s_2,a_2,r_2,\cdots\}$
  - 根据收集到的数据和价值网络估计每个状态动作对的优势$A(s_t,a_t)$
  - 计算策略目标的梯度g
  - 用共轭梯度算法计算$x=H^{-1}g$
  - 用线性搜索找到一个$i$值，并更新策略网络参数$\theta_{k+1}=\theta_k+\alpha^i\sqrt{\frac{2\delta}{x^THx}}x$，其中$i\in\{1,2,\dots,K\}$为能提升策略并满足KL距离限制的最小整数
  - 更新价值网络参数$\omega=\omega+\alpha_\omega\sum_t\delta_t\nabla_{\omega}log\pi_\omega(a_t|s_t)$
- **end for**

## 广义优势估计（GAE）

> 时序差分误差
> $$
> \delta_t=r_t+{\gamma}V(s_{t+1})-V(s_t))
> $$
> 将多步优势估计进行指数加权平均：
> $$
> \begin{align*}
> A_t^{GAE}&=(1-\lambda)(A_t^{(1)}+{\lambda}A_t^{(2)}+{\lambda}^2A_t^{(3)}+\cdots)\\
> &\vdots\\
> &=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
> \end{align*}
> $$
> 其中，$\lambda\in[0,1]$​是额外引入的超参数

代码：

``````python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
``````

## 代码实践

### 库

```python
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy
```

### 网络和算法

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPO:
    """TRPO算法"""

    def __init__(self, hidden_dim, state_space, action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]

        action_dim = action_space.n
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector.vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 共轭梯度法求x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        Hd = self.hession_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists,
                                    descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['new_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

```



# PPO

> $J_{PPO}^{\theta'}=J^{\theta'}(\theta)-{\beta}KL(\theta,\theta')$             ${\nabla}f(x)=f(x){\nabla}logf(x)$
> $$
> J^{\theta'}(\theta)=E_{(s_t,a_t)\thicksim{\pi_{\theta'}}}[\frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}A^{\theta'}(s_t,a_t)]
> $$

> $$
> J_{PPO2}^{\theta'}{\approx}{\sum_{(s_t,a_t)}}min((\frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}A^{\theta^k(s_t,a_t)},clip(\frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)},1-\epsilon,1+\epsilon)A^{\theta^k}(s_t,a_t))
> $$

1. 收集经验

   > 使用当前策略在环境中交互得到一批经验数据。包括state，action，reward等

2. 计算优势估计

   > 对于每一个状态和动作对，计算优势函数的估计值。优势函数衡量采取某种动作相对于平均动作的优势，评估动作的好坏程度

3. 计算优化目标

   > 两种不同的优化目标
   >
   > > $$Clip{\ }Surrogate{\ }Objective$$​
   >
   > 限幅，使得范围在$1-\epsilon\thicksim1+\epsilon$之间，
   >
   > > $$Adaptive{\  }KL {\ }Penalty$$

4. 优化

5. step（）


## 代码实践

### 库

```python
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```



### 网络和算法

```python
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
    def __init__(self, env, ppo_type, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-2,
                 la=0.95, gamma=0.98, beta=3.0, kl_constraint=0.0005, epochs=10, num_episodes=500, eps=0.2):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.env = env
        self.ppo_type = ppo_type
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.la = la  # 用于计算优势函数
        self.beta = beta
        self.kl_constraint = kl_constraint
        self.epochs = epochs  # 一条序列的数据用于训练的轮数
        self.num_episodes = num_episodes
        self.eps = eps
        self.return_list = []

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.la * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
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
                actor_loss = torch.mean(-(ratio * advantage - self.beta * kl_div))  # ppo-惩罚
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
                surrogate1 = ratio * advantage
                surrogate2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage  # ppo-截断
                actor_loss = torch.mean(-torch.min(surrogate1, surrogate2))
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
```

### 图形化

```python
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
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
```

### 结果

> C:\Users\20668\.conda\envs\pytorch\python.exe C:\Users\20668\Desktop\learning\PPO.py 
> Iteration 0:   0%|          | 0/50 [00:00<?, ?it/s]ppo-penalty 3.0
> Iteration 0: 100%|██████████| 50/50 [00:13<00:00,  3.58it/s, episode=50, return=470.700]
> Iteration 1: 100%|██████████| 50/50 [00:23<00:00,  2.15it/s, episode=100, return=500.000]
> Iteration 2: 100%|██████████| 50/50 [00:27<00:00,  1.83it/s, episode=150, return=487.500]
> Iteration 3: 100%|██████████| 50/50 [00:20<00:00,  2.49it/s, episode=200, return=261.200]
> Iteration 4: 100%|██████████| 50/50 [00:16<00:00,  2.98it/s, episode=250, return=239.700]
> Iteration 5: 100%|██████████| 50/50 [00:21<00:00,  2.34it/s, episode=300, return=270.200]
> Iteration 6: 100%|██████████| 50/50 [00:21<00:00,  2.28it/s, episode=350, return=369.800]
> Iteration 7: 100%|██████████| 50/50 [00:25<00:00,  1.92it/s, episode=400, return=500.000]
> Iteration 8: 100%|██████████| 50/50 [00:27<00:00,  1.82it/s, episode=450, return=500.000]
> Iteration 9: 100%|██████████| 50/50 [00:27<00:00,  1.85it/s, episode=500, return=500.000]
> Iteration 0:   0%|          | 0/50 [00:00<?, ?it/s]ppo-clip 3.0
> Iteration 0: 100%|██████████| 50/50 [00:09<00:00,  5.39it/s, episode=50, return=301.400]
> Iteration 1: 100%|██████████| 50/50 [00:13<00:00,  3.61it/s, episode=100, return=373.200]
> Iteration 2: 100%|██████████| 50/50 [00:25<00:00,  1.98it/s, episode=150, return=452.800]
> Iteration 3: 100%|██████████| 50/50 [00:26<00:00,  1.87it/s, episode=200, return=500.000]
> Iteration 4: 100%|██████████| 50/50 [00:26<00:00,  1.87it/s, episode=250, return=474.000]
> Iteration 5: 100%|██████████| 50/50 [00:29<00:00,  1.69it/s, episode=300, return=500.000]
> Iteration 6: 100%|██████████| 50/50 [00:31<00:00,  1.58it/s, episode=350, return=493.800]
> Iteration 7: 100%|██████████| 50/50 [00:31<00:00,  1.57it/s, episode=400, return=462.200]
> Iteration 8: 100%|██████████| 50/50 [00:31<00:00,  1.58it/s, episode=450, return=500.000]
> Iteration 9: 100%|██████████| 50/50 [00:29<00:00,  1.67it/s, episode=500, return=499.100]
>
> ![Figure_1](C:\Users\20668\Desktop\learning\note\picture\Figure_1.png)
>
> ![Figure_2](C:\Users\20668\Desktop\learning\note\picture\Figure_2.png)

# Imitation Learning

- behavior cloning

  - Training:$(s,a)\thicksim\widehat\pi(expert)$​

  - Testing:$（s',a'）\thicksim\pi^*(actor\ cloning\ expert)$

    - $if\ \widehat\pi=\pi^*$,$(s,a)\ and\ (s',a')$来自相同分布

    - $if\ \widehat\pi\ and\ \pi^*$不同，s，s'分布可能非常不同

- 逆强化学习（Inverse Reinforcement Learning，IRL）

  >  从expert的行为推断agent的目标得到奖励函数

  - expert得到$\sum_{n=1}^{N}{R(\widehat\tau_n)}$

  $$
  \widehat\pi\rightarrow\{\widehat\tau_1,\widehat\tau_2,\cdots,\widehat\tau_N\}
  $$

  - 保持$\sum_{n=1}^{N}{R(\widehat\tau_n)}>\sum_{n=1}^{N}{R(\tau)}$​
    - 如何操作保持expert始终为最好的？？？

  - 找到一个以$\sum_{n=1}^{N}{R(\tau)}$为基础的$actor\ \pi$
    $$
    \pi\rightarrow\{\tau_1,\tau_2,\cdots,\tau_N\}
    $$

  - 不断重复
  
    
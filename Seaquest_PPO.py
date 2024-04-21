import torch
import gym
from toolPPO import PPO, Memory
import time


def test():
    env_name = "ALE/Seaquest-v5"
    env = gym.make(env_name, render_mode="human", obs_type="grayscale")
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # 210*160
    action_dim = env.action_space.n  # 18
    max_timesteps = 500
    n_latent_var = 64
    lr = 0.0007
    betas = (0.9, 0.99)
    gamma = 0.99
    K_epochs = 4
    eps = 0.2
    n_episodes = 100
    render = True

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps)  # 初始化一个新的PPO对象

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()[0]
        # print(type(state))
        for t in range(max_timesteps):
            if render:
                env.render()
            action = ppo.policy_old.act(state, memory)

            state, reward, done, is_truncated, _ = env.step(action)
            done = done or is_truncated
            ep_reward += reward
            if done:
                print('Episode:{}\tReward:{}'.format(ep, int(ep_reward)))
                break
    env.close()


if __name__ == "__main__":
    test()

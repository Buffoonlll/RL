import gym
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def test_ppo():
    env_name = "ALE/Seaquest-v5"
    env = gym.make(env_name, obs_type="grayscale")
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, learning_rate=0.0007, verbose=1)
    model.learn(total_timesteps=1e6)
    model.save("Seaquest_PPO")
    del model  # delete trained model to demonstrate loading
    model = PPO.load("Seaquest_PPO",env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()



def test():
    env_name = "ALE/Seaquest-v5"
    env = gym.make(env_name, obs_type="grayscale")
    env = DummyVecEnv([lambda: env])
    loaded_model = PPO.load("Seaquest_PPO", verbose=1)
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=100, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    test_ppo()

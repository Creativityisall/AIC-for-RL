from main import ADM
from baseline import PolicyAgent_18, PolicyAgent_19

def run(env, agent):
    obs = env.reset()
    done = False
    total_reward = 0.0

    total_step = 100
    for step in range(total_step):
        action = agent.act(obs)
        obs, reward, done, info = env.step(obs, action)
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env = ADM(state_dim=5, action_dim=3)
    env.load("./checkpoints/model_adm.pth")
    agent1 = PolicyAgent_18("model_18.pth")
    agent2 = PolicyAgent_19("model_19.pth")
    total_reward1 = 0
    total_reward2 = 0
    total_episodes = 100
    for episode in range(total_episodes):
        total_reward1 += run(env, agent1)
        total_reward2 += run(env, agent2)
    print(f"Average Reward for Agent 18 : {total_reward1 / total_episodes}")
    print(f"Average Reward for Agent 19 : {total_reward2 / total_episodes}")
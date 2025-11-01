from tqdm import tqdm

from main import ADM
from baseline_models.v1.agent import PolicyAgent, MODEL_DIR

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
    env = ADM(state_dim=5, action_dim=3, hidden_dim=516)
    env.load("./checkpoints/model_adm.pth")

    agent = PolicyAgent()
    print("Loading agent from:", MODEL_DIR)
    agent.reset()

    total_reward = 0
    total_episodes = 100
    for episode in tqdm(range(total_episodes), desc="Episodes"):
        total_reward += run(env, agent)
    print(f"Average Reward for Agent 18 : {total_reward / total_episodes}")
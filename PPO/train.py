from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
import torch
import numpy as np
from agent_ppo import Agent, make_graph

# specify agent parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_eps = 100

# create a dummy environment to get node feature dims
env = make("orbit_wars", debug=True)

dummy_obs = env.steps[0][0].observation
dummy_graph = make_graph(obs=dummy_obs.planets)
n_features = dummy_graph.num_node_features

del(env)
# dummy environment deleted, agent created
agent = Agent(num_features=n_features, device=device)

for episode in range(num_eps):
    # reset env
    env = make("orbit_wars", debug=True)
    env.reset()
    obs = env.steps[0][0].observation
    omega = obs.angular_velocity
    n = len(obs.planets)
    ships = torch.zeros(n, dtype=int, device=device)
    done = False
    i = 0
    owned = []
    rewards = []
    accounted_fleets = []
    while not done :
        state_graph = make_graph(obs.planets)
        fleets = [x for x in obs.fleets if x[0] not in accounted_fleets]
        owned = 1-torch.clip(torch.tensor([planet[1] for planet in obs.planets], dtype= int, device=device), min=0, max=1)
        ships = torch.tensor([planet[-2] for planet in obs.planets if planet[1]==0])
        # send input to agent and get output
        weights, log_probs, v = agent.forward(x = state_graph.x, edge_index=state_graph.edge_index, non_graph_state_vars= [obs.remainingOverageTime], ownership_mask=owned)
        actions_formatted = weights * ships

        # give envirnment actions
        stepreturns = env.step(actions = [actions_formatted, actions_formatted])[0]
        obs = stepreturns.observation
        # store rewards
        rewards.append(stepreturns.reward)
        i+=1
    # optimize agent
    agent.optimize()

    del(env)
    # close env

